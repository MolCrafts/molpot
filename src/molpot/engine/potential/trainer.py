from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import torch
from ignite.engine.events import Events, State
from ignite.handlers import ProgressBar, TensorboardLogger, global_step_from_engine
from ignite.metrics import EpochWise, Metric, MetricUsage

import molpot as mpot

from ..base import MolpotEngine
from .utils import create_supervised_evaluator, create_supervised_trainer
from ignite.engine import Engine

logger = mpot.get_logger("molpot.engine")


class PotentialTrainer(MolpotEngine):

    trainer: Engine
    evaluator: Engine

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
        no_grad_eval: bool = False,
        clip_grad_norm: float | None = None,
        work_dir: Path = Path.cwd(),
    ):
        super().__init__(work_dir=work_dir)

        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.non_blocking = non_blocking
        self.deterministic = deterministic
        self.amp_mode = amp_mode
        self.optimizer = optimizer

        self.add_engine(
            "trainer",
            create_supervised_trainer(
                self.model,
                optimizer,
                loss_fn,
                device=device,
                deterministic=deterministic,
                amp_mode=amp_mode,
                scaler=scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                clip_grad_norm=clip_grad_norm,
            ),
        )
        self.add_engine(
            "evaluator",
            create_supervised_evaluator(
                self.model,
                device=self.device,
                amp_mode=self.amp_mode,
                no_grad=no_grad_eval,
            ),
        )

        self.loggers = {}

        # metrics settings
        self._metrics = defaultdict(dict)
        self._metrics_usage = {}

    def compile(self):
        self.model = self.model.to(self.device)
        self.loss_fn = self.loss_fn.to(self.device)
        self.model = torch.compile(
            self.model, dynamic=True, fullgraph=True, mode="reduce-overhead"
        )

    def add_lr_scheduler(self, scheduler):
        from ignite.handlers import LRScheduler

        scheduler_handler = LRScheduler(scheduler)
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=10000), scheduler_handler
        )

    def add_lw_scheduler(self, scheduler):

        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=1000), lambda: scheduler.step()
        )

    def add_checkpoint(
        self,
        save_dir,
        n_every=int(1e5),
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
        save_dir = self.get_absolute_path(save_dir)
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
        max_steps = int(max_steps) if max_steps is not None else None
        max_epochs = int(max_epochs) if max_epochs is not None else None
        epoch_length = int(epoch_length) if epoch_length is not None else None
        if eval_data is not None:
            self.trainer.add_event_handler(
                Events.EPOCH_COMPLETED,
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

    def set_metric_usage(self, **engine_metric: dict[str, MetricUsage]) -> None:
        for engine, usage in engine_metric.items():
            self._metrics_usage[engine] = usage

    def add_metric(
        self,
        name: str,
        metric_factory_fn: Callable[[], Metric],
        engine: str | list[str] | None = None,
    ) -> None:
        metric = metric_factory_fn()

        if isinstance(engine, str):
            _engine = [engine]
        elif engine is None:
            _engine = self._engines.keys()
        else:
            raise ValueError("engine must be a string or a list of strings")

        for engine_name in _engine:
            usage = self._metrics_usage.get(engine_name, None)
            metric.attach(self._engines[engine_name], name, usage)
            self._metrics[engine_name][name] = metric

    def enable_progressbar(self, engine: str) -> None:
        ProgressBar().attach(self._engines[engine])

    def attach_tensorboard(self, log_dir: str):
        log_dir = self.get_absolute_path(log_dir)
        tb_logger = TensorboardLogger(log_dir)

        # add default training loss
        tb_logger.attach_output_handler(
            self.trainer,
            event_name=self._metrics_usage["trainer"].COMPLETED,
            tag="trainer",
            output_transform=lambda x: {"loss": x["loss"]},
            global_step_transform=global_step_from_engine(self.trainer),
        )
        for engine_name, metric_info in self._metrics.items():
            tb_logger.attach_output_handler(
                self._engines[engine_name],
                event_name=self._metrics_usage[engine_name].COMPLETED,
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

    def run_tensorboard_profiler(
        self, loader, wait=1, warmup=1, active=3, repeat=1, log_dir="."
    ):
        logger.info("Running Tensorboard Profiler")
        log_dir = self.get_absolute_path(log_dir)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            self.trainer.add_event_handler(
                Events.ITERATION_STARTED,
                lambda engine: (
                    engine.terminate()
                    if engine.state.iteration >= wait + warmup + active
                    else None
                ),
            )
            self.trainer.add_event_handler(
                Events.ITERATION_COMPLETED, lambda engine: prof.step()
            )

            self.trainer.run(loader, max_epochs=torch.inf)

        prof.export_chrome_trace(log_dir + "/profiler_trace.json")
        logger.info("Profiler run completed")

    def reset(self):
        for engine in self._engines.values():
            engine.state = State()
