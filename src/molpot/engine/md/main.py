from tkinter.tix import MAIN
import torch
import torch.nn as nn
from contextlib import nullcontext
from ignite.engine import Engine, EventEnum, Events

class MDMainEvents(EventEnum):
    """
    timestep execution events. The protocol is adapted from: https://docs.lammps.org/Developer_flow.html
    """
    PRE_FORCE = "pre_force"
    POST_FORCE = "post_force"
    POST_INTEGRATE = "post_integrate"
    FINAL_INTEGRATE = "final_integrate"
    END_STEP = "end_step"
    INITIAL_INTEGRATE = "initial_integrate"
    PRE_NEIGHBOR = "pre_neighbor"
    POST_NEIGHBOR = "post_neighbor"

def main_process(integrator: nn.Module, calculator: nn.Module, neighborlist: nn.Module, gradients_required):

    calculator.eval()

    # Check, if computational graph should be built
    if gradients_required:
        grad_context = torch.no_grad()
    else:
        grad_context = nullcontext()


    def update(engine: Engine, frame):
        with grad_context:

            # Do half step momenta
            engine.fire_event(MDMainEvents.INITIAL_INTEGRATE)
            integrator.half_step(frame)
            # Do propagation MD/PIMD
            engine.fire_event(MDMainEvents.POST_INTEGRATE)
            integrator.main_step(frame)

            engine.fire_event(MDMainEvents.PRE_NEIGHBOR)
            neighborlist(frame)
            engine.fire_event(MDMainEvents.POST_NEIGHBOR)

            engine.fire_event(MDMainEvents.PRE_FORCE)
            calculator(frame)
            engine.fire_event(MDMainEvents.POST_FORCE)

            # Do half step momenta
            integrator.half_step(frame)
            engine.fire_event(MDMainEvents.FINAL_INTEGRATE)

            engine.fire_event(MDMainEvents.END_STEP)

        return frame


    return update