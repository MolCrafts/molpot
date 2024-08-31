# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2023-12-07
# version: 0.0.1

import logging

import numpy as np

from molpot.engine.fix.base import Fix
from molpot.engine.base import Engine


class EarlyStop(Fix):

    def __init__(self, key, patience=5, min_delta=0):
        super().__init__()
        self.key = key
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf

    def __call__(self, trainer: Engine, status: dict, inputs: dict):

        val_loss = status['status'][self.key]
        if self.counter < self.patience:
            delta = val_loss - self.best_loss
            if delta > self.min_delta:
                self.counter += 1
            else:
                self.counter = 0
                self.best_loss = min(val_loss, self.best_loss)

        else:
            status = trainer.Status.STOPPING


class StepCounter(Fix):

    def __init__(self, stop_step: int):
        super().__init__()
        self.stop_step = stop_step

    def __call__(self, trainer: Engine, status: dict, inputs: dict):
        if status["current_step"] > self.stop_step:
            status["flag"] = trainer.Status.FINISHED


class EpochCounter(Fix):

    def __init__(self):
        super().__init__()

    def __call__(self, trainer: Engine, status: dict, inputs: dict):
        if status["current_epoch"] > status["epoch_to_run"]:
            status["flag"] = trainer.Status.FINISHED
