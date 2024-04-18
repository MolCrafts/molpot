# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2023-12-07
# version: 0.0.1

import logging

import numpy as np

from ..fix import Fix


class Strategy(Fix):
    pass

class EarlyStop(Strategy):

    def __init__(self, key, patience=5, min_delta=0):
        super().__init__(0, 0)
        self.key = key
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

        
    def after_iter(self) -> None:

        val_loss = self.trainer.train_outputs[self.key]
        if self.counter < self.patience:
            delta = val_loss - self.best_loss
            if delta > self.min_delta:
                self.counter +=1
            else:
                self.counter = 0
                self.best_loss = min(val_loss, self.best_loss)                

        else:
            logging.info(f"Reach the max patience {self.patience}")
            self.trainer.status = self.trainer.Status.STOP_TRAIN

class StepCounter(Strategy):

    def __init__(self, ):
        super().__init__(0, 0)
        
    def before_iter(self) -> bool:
        if self.trainer.elasped_steps > self.trainer.train_steps:
            self.trainer.status = self.trainer.Status.STOP_TRAIN