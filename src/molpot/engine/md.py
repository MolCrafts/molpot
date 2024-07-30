import torch
from enum import IntEnum
from ..potential.base import Potential
from .fix import FixManager
from .base import Engine


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

    def run(self, system, n_steps):

        status = {}
        inputs = system
        outputs = {}

        if self.progress:
            from tqdm import trange

            iterator = trange
        else:
            iterator = range

        if self.gradients_required:
            grad_context = torch.no_grad()
        else:
            grad_context = torch.enable_grad()

        with grad_context:

            # perform init computation of forces
            inputs, outputs = self.potential(inputs, outputs)

            self.before_run(status, inputs, outputs)
            self.fix.apply(self.Stage.before_run, self, status, inputs, outputs)
            if status['flag'] > self.Status.STOPPING:
                return system

            for step in iterator(n_steps):

                self.before_step(status, inputs, outputs)
                self.fix.apply(self.Stage.before_step, self, status, inputs, outputs)
                if status['flag'] > self.Status.STOPPING:
                    break

                self.half_step(status, inputs, outputs)
                self.fix.apply(self.Stage.half_step, self, status, inputs, outputs)
                if status['flag'] > self.Status.STOPPING:
                    break

                self.main_step(status, inputs, outputs)
                self.fix.apply(self.Stage.main_step, self, status, inputs, outputs)
                if status['flag'] > self.Status.STOPPING:
                    break

                self.after_step(status, inputs, outputs)
                self.fix.apply(self.Stage.after_step, self, status, inputs, outputs)
                if status['flag'] > self.Status.STOPPING:
                    break

                status['current_step'] += 1

            self.after_run(status, inputs, outputs)
            self.fix.apply(self.Stage.after_run, self, status, inputs, outputs)

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

