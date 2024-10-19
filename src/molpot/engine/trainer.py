from .base import MolpotEngine
import torch
from typing import Any, Callable, Union, Iterable, Sequence, Optional, Mapping, Tuple
import collections
from ignite.metrics import Metric
from ignite.engine import (
    create_supervised_trainer,
    create_supervised_evaluator,
    _prepare_batch,
)
from ignite.utils import convert_tensor
from ignite.engine.events import (
    CallableEventWithFilter,
    EventEnum,
    Events,
    EventsList,
    RemovableEventHandle,
    State,
)

def _prepare_batch(
    batch: Sequence[torch.Tensor], device: Optional[Union[str, torch.device]] = None, non_blocking: bool = False
) -> Tuple[Union[torch.Tensor, Sequence, Mapping, str, bytes], ...]:
    """Prepare batch for training or evaluation: pass to a device with options."""
    batch.to(device=device, non_blocking=non_blocking) if device is not None else batch
    return (
        batch,
        batch
    )


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
