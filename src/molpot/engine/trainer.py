import collections
from typing import (Any, Callable, Iterable, Mapping, Optional, Sequence,
                    Tuple, Union)

import torch
from ignite.engine import (_prepare_batch, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.engine.events import (CallableEventWithFilter, EventEnum, Events,
                                  EventsList, RemovableEventHandle, State)
from ignite.handlers import Checkpoint, ProgressBar, global_step_from_engine
from ignite.metrics import EpochWise, Metric, MetricUsage
from ignite.utils import convert_tensor

from .base import MolpotEngine


def _prepare_batch(
    batch: Sequence[torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    batch.to(device=device, non_blocking=non_blocking) if device is not None else batch
    return (batch, batch)


class PotentialTrainer(MolpotEngine):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable[[Any, Any], torch.Tensor], torch.nn.Module],
        device: Union[str, torch.device] | None = None,
        non_blocking: bool = False,
        prepare_batch: Callable = _prepare_batch,
        model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[
            [Any, Any, Any, torch.Tensor], Any
        ] = lambda x, y, y_pred, loss: loss.item(),
        deterministic: bool = False,
        amp_mode: str | None = None,
        scaler: Union[bool, "torch.cuda.amp.GradScaler"] = False,
        gradient_accumulation_steps: int = 1,
        model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    ):
        super().__init__()

        self.model = model
        self.device = device
        self.non_blocking = non_blocking
        self.prepare_batch = prepare_batch
        self.model_transform = model_transform
        self.output_transform = output_transform
        self.deterministic = deterministic
        self.amp_mode = amp_mode
        self.model_fn = model_fn
        self.optimizer = optimizer

        self.add_engine(
            "trainer",
            create_supervised_trainer(
                model,
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
        metrics: dict[str, Metric] = None,
    ) -> None:
        self.add_engine(
            "evaluator",
            create_supervised_evaluator(
                self.model,
                metrics=metrics,
                device=self.device,
                non_blocking=self.non_blocking,
                prepare_batch=self.prepare_batch,
                model_transform=self.model_transform,
                output_transform=self.output_transform,
                amp_mode=self.amp_mode,
                model_fn=self.model_fn,
            ),
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
        self.trainer.run(data, max_epochs=max_epochs, epoch_length=epoch_length)

    def add_metric(self, name: str, metric: Metric, target: str, usage: str | MetricUsage = EpochWise()) -> None:
        metric.attach(self._engines[target], name, usage)

    def enable_progressbar(self, target: str="trainer") -> None:
        ProgressBar().attach(self._engines[target])