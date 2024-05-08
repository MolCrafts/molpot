import time

import torch

from ..fix import Fix
from typing import Callable
from collections import defaultdict
from molpot.statistic import Tracker

from copy import deepcopy


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


class Validater:

    def __init__(self, model, loss_fn, valid_dataset):
        self.model = model
        self.loss_fn = loss_fn
        self.valid_dataset = valid_dataset
        self.metrics = {}
        self.metrics_list = defaultdict(list)
        self.fixes = []

    def validate(self, steps):
        self.steps = steps
        self.model.eval()
        for self.data in self.valid_dataset:
            self.outputs = self.model(self.data)
            loss = self.loss_fn(self.outputs, self.data)
            self.metrics["loss"] = loss.item()
            for fix in self.fixes:
                fix.after_iter()
            for key, value in self.metrics.items():
                self.metrics_list[key].append(value)

        return {
            key: torch.mean(torch.tensor(value))
            for key, value in self.metrics_list.items()
        }


class ValidateFix(Fix):

    def __init__(self, every_n_steps: int, validate_dataset, every_n_epochs: int = 0):
        self.name = "validate"
        super().__init__(every_n_steps, every_n_epochs)
        self.priority = 0

        self.validate_dataset = validate_dataset

    def before_train(self) -> None:
        self.trainer.valid_results = {}
        self.validater = Validater(
            self.trainer.model, self.trainer.loss_fn, self.validate_dataset
        )
        for fix in [fix for fix in self.trainer.fixes if isinstance(fix, Metric)]:
            new_fix = fix.copy()
            new_fix.trainer = self.validater
            self.validater.fixes.append(new_fix)

    def after_iter(self) -> None:

        self.trainer.model.eval()
        result = self.validater.validate(self.trainer.steps)
        self.trainer.valid_results.update(result)
        self.trainer.model.train()
