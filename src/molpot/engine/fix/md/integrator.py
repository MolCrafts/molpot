import torch.nn as nn
from ...md import MDEngine
from ..base import Fix
from typing import Callable

class Integrator(Fix):

    def __init__(self, time_step: float):
        super().__init__(priority=1)
        self.time_step = time_step


class NVE(Integrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    def __init__(self, timestep: float, wrap_fn: Callable):
        super().__init__(timestep)
        self.wrap_fn = wrap_fn

    def forward(self, engine: MDEngine, status, inputs, outputs):
        match status["stage"]:

            case engine.Stage.main_step:

                outputs["atoms"]["xyz"] = self.wrap_fn(
                    outputs["atoms"]["xyz"]
                    + self.time_step
                    * outputs["atoms"]["momentum"]
                    / inputs["atoms"]["mass"]
                )
                return

            case engine.Stage.half_step:

                outputs["atoms"]["momentum"] = (
                    outputs["atoms"]["momentum"]
                    + 0.5 * self.time_step * outputs["atoms"]["forces"]
                )
                return
