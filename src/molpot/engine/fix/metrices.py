import time

import torch

from .base import Fix
from molpot.statistic import Tracker


class MAE(Fix):
    def __init__(
        self, output_key: str, every_n_steps: int, result_key, target_key, reduction="mean"
    ):
        self.output_key = output_key
        super().__init__()
        self.result_key = result_key
        self.target_key = target_key
        self.every_n_steps = every_n_steps
        self.kernel = torch.nn.L1Loss(reduction=reduction)

    def __call__(self, trainer, status, inputs) -> None:

        step = status["current_step"]
        if step % self.every_n_steps == 0:
            result = inputs[self.result_key]
            target = inputs[self.target_key]
            mae = self.kernel(result, target)
            status['metrices'][self.output_key] = mae


class StepSpeed(Fix):

    def __init__(self, every_n_steps: int):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.start_time = time.perf_counter()
        self.tracker = Tracker()

    def __call__(self, trainer, status, inputs) -> None:
        if status["current_step"] % self.every_n_steps == 0:
            end_time = time.perf_counter()
            iter_time = end_time - self.start_time
            self.tracker.update(iter_time / self.every_n_steps)
            status['metrices']["step_speed"] = self.tracker.mean
            self.start_time = end_time


class EpochSpeed(Fix):

    def __init__(self, every_n_epochs: int):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.start_time = time.perf_counter()
        self.tracker = Tracker()

    def __call__(self, trainer, status, inputs) -> None:
        if status["current_step"] % self.every_n_epochs == 0:
            end_time = time.perf_counter()
            iter_time = end_time - self.start_time
            self.tracker.update(iter_time / self.every_n_epochs)
            status['metrices']["epoch_speed"] = self.tracker.mean
            self.start_time = end_time
