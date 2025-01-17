from functools import partial
from typing import Any, Callable, Iterable, Union

import torch
from ignite.engine.events import EventEnum, Events, State
from ignite.handlers import (
    ProgressBar,
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.metrics import EpochWise, Metric, MetricUsage

from ..base import MolpotEngine
from .utils import (
    create_supervised_trainer,
    create_supervised_evaluator,
)
from collections import defaultdict


class PotentialTrainer(MolpotEngine):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
        device: Union[str, torch.device] | None = None,
        non_blocking: bool = False,
        deterministic: bool = False,
        amp_mode: str | None = None,
        scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
        gradient_accumulation_steps: int = 1,
        model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
        no_grad_eval: bool = False,
    ):
        super().__init__()

        self.model = model
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.loss_fn = loss_fn
        self.device = device
        self.non_blocking = non_blocking
        # self.output_transform = output_transform
        self.deterministic = deterministic
        self.amp_mode = amp_mode
        self.model_fn = model_fn
        self.optimizer = optimizer

        self.add_engine(
            "trainer",
            create_supervised_trainer(
                self.model,
                optimizer,
                loss_fn,
                device=device,
                non_blocking=non_blocking,
                deterministic=deterministic,
                amp_mode=amp_mode,
                scaler=scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                model_fn=model_fn,
            ),
        )
        self.add_engine(
            "evaluator",
            create_supervised_evaluator(
                self.model,
                device=self.device,
                non_blocking=self.non_blocking,
                amp_mode=self.amp_mode,
                model_fn=self.model_fn,
                no_grad=no_grad_eval,
            ),
        )
        
        self.metrics = defaultdict(dict)
        self.loggers = {}
    def compile(self):
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)

    @property
    def trainer(self):
        return self._engines["trainer"]

    @property
    def evaluator(self):
        if "evaluator" not in self._engines:
            raise ValueError("Evaluator not yet added to the trainer")
        return self._engines["evaluator"]

    def add_lr_scheduler(self, scheduler):
        from ignite.handlers import LRScheduler
        scheduler_handler = LRScheduler(scheduler)
        self.trainer.add_event_handler(Events.ITERATION_STARTED, scheduler_handler)

    def add_checkpoint(
        self,
        save_dir,
        n_every=None,
        n_epoch=None,
        filename_prefix="",
        score_function=None,
        score_name=None,
        n_saved=1,
        global_step_transform=None,
        filename_pattern=None,
        include_self=False,
        greater_or_equal=False,
        save_on_rank=0,
    ):
        from ignite.handlers import Checkpoint

        to_save = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer": self.trainer,
        }
        # save_handler =
        if n_every is not None:
            prefix = f"{filename_prefix}_step"
            handler = Checkpoint(
                to_save,
                save_dir,
                filename_prefix=prefix,
                score_function=score_function,
                score_name=score_name,
                n_saved=n_saved,
                global_step_transform=global_step_transform,
                filename_pattern=filename_pattern,
                include_self=include_self,
                greater_or_equal=greater_or_equal,
                save_on_rank=save_on_rank,
            )
            self.trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every=n_every), handler
            )

        if n_epoch is not None:
            prefix = f"{filename_prefix}_epoch"
            handler = Checkpoint(
                to_save,
                save_dir,
                filename_prefix=filename_prefix,
                score_function=score_function,
                score_name=score_name,
                n_saved=n_saved,
                global_step_transform=global_step_transform,
                filename_pattern=filename_pattern,
                include_self=include_self,
                greater_or_equal=greater_or_equal,
                save_on_rank=save_on_rank,
            )
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=n_epoch), handler
            )

    def run(
        self,
        train_data: Iterable | None = None,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        eval_data: Iterable | None = None,
        epoch_length: int | None = None,
    ) -> State:

        if eval_data is not None:
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=1),
                lambda: self.evaluator.run(
                    eval_data, epoch_length=epoch_length, max_epochs=1
                ),
            )
        if max_steps is not None:
            self.trainer.add_event_handler(
                Events.ITERATION_STARTED,
                lambda engine: (
                    engine.terminate() if engine.state.iteration >= max_steps else None
                ),
            )
            max_epochs = torch.inf  # TODO: inferring max_epochs from max_steps

        state = self.trainer.run(
            train_data, max_epochs=max_epochs, epoch_length=epoch_length
        )

        for logger in self.loggers.values():
            logger.close()
        return state

    def add_metric(
        self,
        name: str,
        metric: Metric,
        usage: str | MetricUsage = EpochWise(),
        engine: str = "trainer",
    ) -> None:
        metric.attach(self._engines[engine], name, usage)
        self.metrics[engine][name] = metric

    def enable_progressbar(self, engine: str) -> None:
        ProgressBar().attach(self._engines[engine])

    def attach_tensorboard(
        self,
        log_dir: str,
        tag_event_map = {
            "trainer": Events.ITERATION_COMPLETED(every=100),
            "evaluator": Events.EPOCH_COMPLETED,
        }
    ):
        
        tb_logger = TensorboardLogger(log_dir)
        # add default training loss
        tb_logger.attach_output_handler(
            self.trainer,
            event_name=tag_event_map["trainer"],
            tag="trainer",
            output_transform=lambda x: {"loss": x["loss"]},
            global_step_transform=global_step_from_engine(self.trainer),
        )
        for engine_name, metric_info in self.metrics.items():
                tb_logger.attach_output_handler(
                self._engines[engine_name],
                event_name=tag_event_map[engine_name],
                tag=engine_name,
                metric_names=list(metric_info.keys()),
                global_step_transform=lambda e, event: self.trainer.state.iteration,
            )

        self.loggers["tensorboard"] = tb_logger

    # def attach_basic_time_profiler(self):

    #     basic_profiler = BasicTimeProfiler()
    #     basic_profiler.attach(self.trainer)
    #     return basic_profiler

    # def attach_handlers_time_profiler(self):
    #     handlers_profiler = HandlersTimeProfiler()
    #     handlers_profiler.attach(self.trainer)
    #     return handlers_profiler

    # def run_model_profiler(self, backend: str = "tensorboard"):

    #     match backend:
    #         case "tensorboard":
    #             self.run_tensorboard_profiler()

    # def run_tensorboard_profiler(
    #     self, loader, wait=1, warmup=1, active=3, repeat=1, log_dir="."
    # ):

    #     process_fn = self.trainer._process_function

    #     with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CUDA,
    #             torch.profiler.ProfilerActivity.CPU,
    #         ],
    #         schedule=torch.profiler.schedule(
    #             wait=wait, warmup=warmup, active=active, repeat=repeat
    #         ),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #     ) as prof:
    #         for step, batch in enumerate(loader):
    #             if step >= wait + warmup + active:
    #                 break
    #             self.optimizer.zero_grad()
    #             self.model.train()
    #             x, y = self.prepare_batch(
    #                 batch, device=self.device, non_blocking=self.non_blocking
    #             )
    #             output = self.model_fn(self.model, x)
    #             y_pred = model_transform(output)
    #             loss = self.loss_fn(y_pred, y)
    #             loss.backward()
    #             self.optimizer.step()
    #             prof.step()

    def reset(self):
        for engine in self._engines.values():
            engine.state = State()
