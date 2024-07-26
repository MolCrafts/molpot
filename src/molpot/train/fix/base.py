from __future__ import annotations
from typing import Type, TypeVar
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from molpot.train.trainer import Trainer


class Fix(ABC):

    def __init__(self, priority: int = 5):
        assert priority >= 0 and priority < 10
        self.priority = priority

    @abstractmethod
    def __call__(self, trainer: Trainer, status:dict, inputs: dict, outputs: dict):
        pass

    def __repr__(self):
        return f"<Fix {self.name}: {self.priority}>"

    @property
    def name(self):
        return self.__class__.__name__


class FixManager:

    def __init__(self, stages: Type[Trainer.Stage]):
        self.fixes: dict[int, list[Fix]] = {}
        for stage in stages:
            self.fixes[stage] = []

    def register(self, stage: Trainer.Stage, fix: Fix):

        assert isinstance(fix, Fix)
        assert stage in self.fixes
        self.fixes[stage].append(fix)
        self.fixes[stage].sort(key=lambda x: x.priority)

    def apply(
        self, which_stage: Trainer.Stage, trainer: Trainer, status:dict, inputs: dict, outputs: dict
    ):
        for fix in self.fixes[which_stage]:
            fix(trainer, status, inputs, outputs)
