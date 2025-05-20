from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import torch
from ignite.handlers import ProgressBar, TensorboardLogger, global_step_from_engine
from ignite.metrics import EpochWise, Metric, MetricUsage

import molpot as mpot
from ignite.engine import Events, State
from ..base import MolpotEngine
from .utils import create_supervised_evaluator, create_supervised_trainer
from ignite.engine import Engine

logger = mpot.get_logger("molpot.engine")
config = mpot.get_config()


class PotentialTrainer(MolpotEngine):
    """
    PotentialTrainer is a specialized engine for training and evaluating potential models.

    This class provides a comprehensive framework for training potential energy models
    with PyTorch, using the Ignite engine underneath. It manages the training loop,
    evaluation, checkpointing, metrics collection, and performance monitoring.

    Attributes:
        trainer (Engine): The Ignite engine for training.
        evaluator (Engine): The Ignite engine for evaluation.
        events (Events): Event types supported by the engines.
        model (torch.nn.Module): The model being trained.
        loss_fn (Callable or torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer for parameter updates.
        loggers (dict): Dictionary of attached loggers.
        
    Methods:
        compile(): Compiles the model for optimized execution.
        add_lr_scheduler(scheduler): Adds a learning rate scheduler.
        add_lw_scheduler(scheduler): Adds a weight scheduler.
        add_checkpoint(save_dir, ...): Sets up model checkpointing.
        run(train_data, ...): Executes the training loop.
        set_metric_usage(**engine_metric): Sets how metrics are used across engines.
        add_metric(name, metric_factory_fn, engine): Adds a metric to track.
        enable_progressbar(engine): Enables a progress bar for the specified engine.
        attach_tensorboard(log_dir): Sets up TensorBoard logging.
        enable_dataset_update(every, condition): Enables periodic dataset updates.
        run_tensorboard_profiler(loader, ...): Runs performance profiling.
        reset(): Resets the state of all engines.
    """

    trainer: Engine
    evaluator: Engine

    events = Events

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module] | None,
        non_blocking: bool = False,
        deterministic: bool = False,
        amp_mode: str | None = None,
        scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
        gradient_accumulation_steps: int = 1,
        no_grad_eval: bool = False,
        clip_grad_norm: float | None = None
    ):
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
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
                device=config.device,
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
                device=config.device,
                amp_mode=self.amp_mode,
                no_grad=no_grad_eval,
            ),
        )

        self.loggers = {}

        # metrics settings
        self._metrics = defaultdict(dict)
        self._metrics_usage = {}

    def compile(self):
        self.model = self.model.to(config.device)
        self.loss_fn = self.loss_fn.to(config.device)
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

    def enable_dataset_update(self, every: int, condition):
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=every),
            lambda engine: (
                self.dataset.update(engine.state.output) if condition else None
            ),
        )

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
