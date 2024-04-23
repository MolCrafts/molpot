import time

import torch

from ..fix import Fix


class Metric(Fix):

    def __init__(self, name: str, every_n_steps: int, every_n_epochs: int = 0):
        super().__init__(every_n_steps, every_n_epochs)
        self.name = name

    def before_train(self):
        self.trainer.metrics[self.name] = 0


class MAE(Metric):
    def __init__(
        self, name: str, every_n_steps: int, result_key, target_key, reduction="mean"
    ):
        self.name = f"{name}_MAE"
        super().__init__(self.name, every_n_steps, every_n_epochs=0)
        self.result_key = result_key
        self.target_key = target_key
        self.kernel = torch.nn.L1Loss(reduction=reduction)

    def after_iter(self) -> None:

        result = self.trainer.train_result[self.result_key]
        target = self.trainer.train_data[self.target_key]
        mae = self.kernel(result, target)
        self.trainer.metrics[self.name] = mae


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
