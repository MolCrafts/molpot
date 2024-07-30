import torch
from .base import Fix

class ClipNorm(Fix):
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def after_iter(self) -> None:
        if self.every_n_steps(self._every_n_steps) or self.is_last_iter():

            if self.trainer.enable_amp:
                self.trainer._grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.trainer.model.parameters(), self.max_norm
            )