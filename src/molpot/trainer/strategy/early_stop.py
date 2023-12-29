# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2023-12-07
# version: 0.0.1

from .base import Strategy
import numpy as np

__all__ = ["Stagnation", "StepCounter"]

class Stagnation(Strategy):

    def __init__(self, key, patience=5, min_delta=0):
        self.key = key
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, step:int, output, data) -> bool:
        val_loss = output[self.key].mean()
        if self.counter < self.patience:
            delta = val_loss - self.best_loss
            if delta > self.min_delta:
                self.counter +=1
            else:
                self.counter = 0
                self.best_loss = min(val_loss, self.best_loss)
            return False
        else:
            return True


class StepCounter(Strategy):

    def __init__(self, nstep:int):
        super().__init__()
        self.nstep = nstep

    def __call__(self, step:int, *args, **kwargs) -> bool:
        if step >= self.nstep:
            return True
        return False