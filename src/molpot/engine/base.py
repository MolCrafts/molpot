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

    @abstractmethod
    def save_ckpt(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def load_ckpt(self, path: str | Path) -> None:
        pass

