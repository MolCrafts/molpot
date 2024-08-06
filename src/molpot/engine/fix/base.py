from __future__ import annotations
import torch.nn as nn
from typing import Type

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..base import Engine


class Fix(nn.Module):

    def __init__(self, priority: int = 5):
        super().__init__()
        assert priority >= 0
        self.priority = priority

    def forward(self, engine: Engine, status: dict, inputs: dict, outputs: dict):
        pass

    def __repr__(self):
        return f"<Fix {self.name}: {self.priority}>"

    @property
    def name(self):
        return self.__class__.__name__

    def state_dict(self) -> dict:
        return {
            'input': {},
            'state': {},
        }
    
    def finalize(self, engine: Engine, status: dict, inputs: dict, outputs: dict):
        pass


class FixManager:

    def __init__(self, stages: Type[Engine.Stage]):
        self.fixes: dict[int, list[Fix]] = {}
        for stage in stages:
            self.fixes[stage] = []

    def register(self, fix: Fix, *stages: Engine.Stage, ):
        for stage in stages:
            assert stage in self.fixes
            self.fixes[stage].append(fix)
            self.fixes[stage].sort(key=lambda x: x.priority, reverse=True)

    def apply(
        self,
        which_stage: Engine.Stage,
        engine: Engine,
        status: dict,
        inputs: dict,
        outputs: dict,
    ):
        # switch engine's status
        status['stage'] = which_stage
        for fix in self.fixes[which_stage]:
            fix(engine, status, inputs, outputs)

    def __repr__(self):
        return f"<FixManager: {self.fixes}>"

    def state_dict(self) -> dict:
        data = {}
        for stage_name, stage in self.fixes.items():
            data[stage_name] = {}
            for fix_name, fix in stage.items():
                data[stage_name][fix_name] = fix.state_dict()

        return data

    def finalize(self,        engine: Engine,
        status: dict,
        inputs: dict,
        outputs: dict):
        for stage in self.fixes.values():
            for fix in stage:
                fix.finalize(engine, status, inputs, outputs)