from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path

from .fix import FixManager


class Engine(ABC):

    class Stage(IntEnum):
        pass

    class Status(IntEnum):

        INIT = 0
        TRAINING = 1

        STOPPING = 2
        FINISHED = 3
        ERROR = 4

    def __init__(self):
        self._fix = FixManager(self.Stage)

    @property
    def fix(self) -> FixManager:
        return self._fix

    def get_status(self):
        return {
            'status': list(self.Status)[0],
            'stage': list(self.Stage)[0],
            'current_step': 0,
            'current_epoch': 0
        }