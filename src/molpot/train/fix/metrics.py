import time

import torch

from .base import Fix

class Metric(Fix):
    pass

class MAE(Metric):
    def __init__(
        self, name: str, every_n_steps: int, result_key, target_key, reduction="mean"
    ):
        name = f"{name}_MAE"
        super().__init__(name, every_n_steps, every_n_epochs=0)
        self.result_key = result_key
        self.target_key = target_key
        self.kernel = torch.nn.L1Loss(reduction=reduction)

    def after_iter(self) -> None:

        result = self.trainer.outputs[self.result_key]
        target = self.trainer.data[self.target_key]
        mae = self.kernel(result, target)
        self.trainer.metrics[self.name] = mae


class Intermidiate(Metric):

    def __init__(self, name: str, every_n_steps: int, every_n_epochs: int = 0):
        super().__init__(name, every_n_steps, every_n_epochs)

    def after_iter(self) -> None:
        self.trainer.metrics[self.name] = self.trainer.outputs[self.name]


class StepSpeed(Metric):

    def __init__(self, every_n_step: int):
        self.name = "step_speed"
        super().__init__(self.name, every_n_step, 0)

    def before_iter(self) -> None:
        self._iter_start_time = time.perf_counter()

    def after_iter(self) -> None:
        iter_time = time.perf_counter() - self._iter_start_time
        self.trainer.metrics[self.name] = iter_time / self.every_n_steps


class EpochSpeed(Metric):

    def __init__(self, every_n_epoch: int):
        self.name = "epoch_speed"
        super().__init__(self.name, every_n_epoch, 0)

    def before_epoch(self) -> None:
        self._iter_start_time = time.perf_counter()

    def after_epoch(self) -> None:
        iter_time = time.perf_counter() - self._iter_start_time
        self.trainer.metrics[self.name] = iter_time / self.every_n_epochs
