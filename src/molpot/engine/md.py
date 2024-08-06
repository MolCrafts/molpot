from pathlib import Path
import torch
from enum import IntEnum
from .base import Engine
import molpot as mpot
from molpot import alias


class MDEngine(Engine):

    class Stage(IntEnum):

        before_run = 0
        before_step = 1
        half_step = 2
        main_step = 3
        after_step = 4
        after_run = 5

    class Status(IntEnum):

        INIT = 0
        TRAINING = 1

        STOPPING = 2
        FINISHED = 3
        ERROR = 4

    def __init__(
        self,
        gradients_required: bool = False,
        progress: bool = True,
    ):
        super().__init__()
        self.gradients_required = gradients_required
        self.progress = progress

    def run(self, system: mpot.System, n_steps):

        status = self.get_status()
        inputs = system.frame
        outputs = mpot.Frame()
        outputs["atoms"]["xyz"] = inputs["atoms"]["xyz"]
        outputs["atoms"]["energy"] = 0
        outputs["atoms"]["forces"] = 0
        outputs["atoms"]["momentum"] = 0
        potential = system.forcefield.get_potential()

        if self.progress:
            from tqdm import trange

            iterator = trange
        else:
            iterator = range

        if self.gradients_required:
            grad_context = torch.enable_grad()
        else:
            grad_context = torch.no_grad()

        with grad_context:

            self.before_run(status, inputs, outputs)
            self.fix.apply(self.Stage.before_run, self, status, inputs, outputs)

            for step in iterator(n_steps):

                self.before_step(status, inputs, outputs)
                self.fix.apply(self.Stage.before_step, self, status, inputs, outputs)

                with torch.enable_grad():
                    inputs, outputs = potential(inputs, outputs)

                self.half_step(status, inputs, outputs)
                self.fix.apply(self.Stage.half_step, self, status, inputs, outputs)

                self.main_step(status, inputs, outputs)
                self.fix.apply(self.Stage.main_step, self, status, inputs, outputs)

                self.after_step(status, inputs, outputs)
                self.fix.apply(self.Stage.after_step, self, status, inputs, outputs)

                status['current_step'] += 1

            self.after_run(status, inputs, outputs)
            self.fix.apply(self.Stage.after_run, self, status, inputs, outputs)
            if status['status'] > self.Status.STOPPING:
                return system

        self.fix.finalize(self, status, inputs, outputs)

        system.frame["atoms"]["xyz"] = outputs["atoms"]["xyz"]
        system.frame["atoms"]["energy"] = outputs["atoms"]["energy"]
        system.frame["atoms"]["forces"] = outputs["atoms"]["forces"]
        system.frame["atoms"]["momentum"] = outputs["atoms"]["momentum"]
        
        return system

    def before_run(self, status, inputs, outputs):
        pass

    def before_step(self, status, inputs, outputs):
        pass

    def half_step(self, status, inputs, outputs):
        pass

    def main_step(self, status, inputs, outputs):
        pass

    def after_step(self, status, inputs, outputs):
        pass

    def after_run(self, status, inputs, outputs):
        pass

