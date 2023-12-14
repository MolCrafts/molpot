# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2023-12-07
# version: 0.0.1

from .base import Strategy
import numpy as np

class EarlyStop(Strategy):

    def __init__(self, patience=5, min_delta=0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, val_loss) -> bool:

        if self.counter < self.patience:
            if val_loss - self.best_loss > self.min_delta:
                self.counter +=1
            else:
                self.counter = 0
                self.best_loss = val_loss
            return False
        else:
            return True
