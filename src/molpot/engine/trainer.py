import collections
from functools import partial
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from ignite.engine import create_supervised_evaluator, create_supervised_trainer
from ignite.engine.events import (
    CallableEventWithFilter,
    EventEnum,
    Events,
    EventsList,
    RemovableEventHandle,
    State,
)
from ignite.handlers import (
    BasicTimeProfiler,
    Checkpoint,
    HandlersTimeProfiler,
    ProgressBar,
    TensorboardLogger,
    Timer,
    global_step_from_engine,
)
from ignite.metrics import EpochWise, Metric, MetricUsage
from ignite.utils import convert_tensor
from tensordict import TensorDict

from .base import MolpotEngine


def convert_tensordict(
    x: TensorDict,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> TensorDict:
    """Move tensors to relevant device.

    Args:
        x: input tensor or mapping, or sequence of tensors.
        device: device type to move ``x``.
        non_blocking: convert a CPU Tensor with pinned memory to a CUDA Tensor
            asynchronously with respect to the host if possible
    """

    return x.to(device=device, non_blocking=non_blocking)


def _prepare_batch(
    batch: TensorDict,
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[TensorDict, TensorDict]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    return (
        convert_tensordict(batch, device=device, non_blocking=non_blocking),  # data
        convert_tensordict(
            batch["labels"], device=device, non_blocking=non_blocking
        ),  # y
    )


def trainer_output_transform(x, y, y_pred, loss):
    return (y_pred, y, loss)


def eval_output_transform(x, y, y_pred):
    return (y_pred, y)


def model_transform(output):
    return output["predicts"]


class PotentialTrainer(MolpotEngine):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
        device: Union[str, torch.device] | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = _prepare_batch,
        model_transform: Callable[[Any], Any] = model_transform,
        output_transform: Callable[
            [Any, Any, Any, torch.Tensor], Any
        ] = trainer_output_transform,
        deterministic: bool = False,
        amp_mode: str | None = None,
        scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
        gradient_accumulation_steps: int = 1,
        model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    ):
        super().__init__()

        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.non_blocking = non_blocking
        self.prepare_batch = partial(
            prepare_batch, device=device, non_blocking=non_blocking
        )
        self.model_transform = model_transform
        self.output_transform = output_transform
        self.deterministic = deterministic
        self.amp_mode = amp_mode
        self.model_fn = model_fn
        self.optimizer = optimizer

        self.metrics = {}
        self.loggers = {}

        self.add_engine(
            "trainer",
            create_supervised_trainer(
                self.model,
                optimizer,
                loss_fn,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                deterministic=deterministic,
                amp_mode=amp_mode,
                scaler=scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                model_fn=model_fn,
            ),
        )

    def add_evaluator(
        self,
        dl=None,
        max_epochs: int | None = None,
        epoch_length: int | None = None,
    ) -> None:
        self.add_engine(
            "evaluator",
            create_supervised_evaluator(
                self.model,
                metrics=None,
                device=self.device,
                non_blocking=self.non_blocking,
                prepare_batch=self.prepare_batch,
                model_transform=self.model_transform,
                output_transform=eval_output_transform,
                amp_mode=self.amp_mode,
                model_fn=self.model_fn,
            ),
        )
        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda trainer: self.evaluator.run(dl, max_epochs, epoch_length),
        )

    @property
    def trainer(self):
        return self._engines["trainer"]

    @property
    def evaluator(self):
        if "evaluator" not in self._engines:
            raise ValueError("Evaluator not yet added to the trainer")
        return self._engines["evaluator"]

    def run(
        self,
        data: Iterable | None = None,
        max_epochs: int | None = None,
        epoch_length: int | None = None,
    ) -> State:
        state = self.trainer.run(data, max_epochs=max_epochs, epoch_length=epoch_length)
        for logger in self.loggers.values():
            logger.close()
        return state

    def add_metric(
        self,
        name: str,
        metric: Metric,
        usage: str | MetricUsage = EpochWise(),
        target: str = "all",
    ) -> None:
        assert isinstance(metric, Metric)
        assert isinstance(usage, MetricUsage)
        assert isinstance(target, str)
        match target:
            case "all":
                for engine in self._engines.values():
                    metric.attach(engine, name, usage)
            case _:
                metric.attach(self._engines[target], name, usage)
        self.metrics[name] = {"metric": metric, "usage": usage}

    def add_handler(
        self, handler, event_name=Events.ITERATION_COMPLETED, target: str = "both"
    ):
        if target == "both":
            for engine in self._engines.values():
                engine.add_event_handler(event_name, handler)
        else:
            self.trainer.add_event_handler(event_name, handler)

    def attach_progressbar(self, target: str = "both") -> None:
        if target == "both":
            for engine in self._engines.values():
                ProgressBar().attach(engine)
        else:
            ProgressBar().attach(self._engines[target])

    def attach_tensorboard(
        self,
        log_dir: str,
        event_name: EventEnum = Events.ITERATION_COMPLETED,
    ):
        tb_logger = TensorboardLogger(log_dir)
        tb_logger.attach_output_handler(
            self.trainer,
            event_name=event_name,
            tag="training",
            output_transform=lambda x: {"loss": x[-1]},
        )
        for ename, engine in self._engines.items():
            tb_logger.attach_output_handler(
                engine,
                event_name,
                tag=ename,
                metric_names=list(self.metrics.keys()),
                global_step_transform=global_step_from_engine(engine),
            )
        self.loggers["tensorboard"] = tb_logger

    def attach_basic_time_profiler(self):

        basic_profiler = BasicTimeProfiler()
        basic_profiler.attach(self.trainer)
        return basic_profiler

    def attach_handlers_time_profiler(self):
        handlers_profiler = HandlersTimeProfiler()
        handlers_profiler.attach(self.trainer)
        return handlers_profiler

    def run_model_profiler(self, backend: str = "tensorboard"):

        match backend:
            case "tensorboard":
                self.run_tensorboard_profiler()

    def run_tensorboard_profiler(
        self, loader, wait=1, warmup=1, active=3, repeat=1, log_dir="."
    ):

        process_fn = self.trainer._process_function

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step, batch in enumerate(loader):
                if step >= wait + warmup + active:
                    break
                self.optimizer.zero_grad()
                self.model.train()
                x, y = self.prepare_batch(
                    batch, device=self.device, non_blocking=self.non_blocking
                )
                output = self.model_fn(self.model, x)
                y_pred = model_transform(output)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                prof.step()
