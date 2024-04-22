import time

import torch

from ..fix import Fix

class Metric(Fix):
    pass

    
class MAE(Metric):
    def __init__(self, every_n_steps:int, result_key, target_key, reduction="mean"):
        super().__init__(every_n_steps, every_n_epochs=0)
        self.result_key = result_key
        self.target_key = target_key
        self.kernel = torch.nn.L1Loss(reduction=reduction)

    def after_iter(self) -> None:
        
        if self.every_n_steps(self._every_n_steps) or self.is_last_iter():
            result = self.trainer.train_result[self.result_key]
            target = self.trainer.train_data[self.target_key]
            loss = self.kernel(result, target)
            self.trainer.metrics["mae"] = loss

class StepSpeed(Metric):

    def __init__(self, every_n_step:int, every_n_epoch:int):
        assert bool(every_n_epoch) ^ bool(every_n_step), "either every_n_step or every_n_epoch should be set."
        super().__init__(every_n_step, every_n_epoch)

    def before_epoch(self) -> None:
        if self._every_n_epochs and self.every_n_epochs(self._every_n_epochs):
            self._epoch_start_time = time.perf_counter()
        
    def after_epoch(self) -> None:
        if self._every_n_epochs and self.every_n_epochs(self._every_n_epochs):
            epoch_time = time.perf_counter() - self._epoch_start_time
            self.trainer.metrics["epoch_time"] = epoch_time / self._every_n_epochs

    def before_iter(self) -> None:
        if self._every_n_steps and self.every_n_steps(self._every_n_steps):
            self._iter_start_time = time.perf_counter()

    def after_iter(self) -> None:
        if self._every_n_steps and self.every_n_steps(self._every_n_steps):
            iter_time = time.perf_counter() - self._iter_start_time
            self.trainer.metrics["iter_time"] = iter_time / self._every_n_steps