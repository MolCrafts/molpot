import torch.nn as nn
from ...md import MDEngine
from ..base import Fix

class Integrator(Fix):
    
    def __init__(self, time_step: float):
        super().__init__()
        self.time_step = time_step

class NVE(Integrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    def __init__(self, timestep: float):
        super().__init__(timestep)

    def forward(self, engine:MDEngine, status, inputs, outputs):

        match status['stage']:

            case engine.Stage.main_step:

                outputs['xyz'] = outputs['xyz'] + self.time_step * outputs['momentum'] / inputs['mass']

            case engine.Stage.half_step:

                outputs['momentum'] = outputs['momentum'] + 0.5 * self.time_step * outputs['force']

            case _:
                raise ValueError(f"Invalid stage {status['stage']} for integrator {self.__class__.__name__}")
            