import torch
import torch.nn as nn
from contextlib import nullcontext
from ignite.engine import Engine
from .events import MDMainEvents

def main_process(gradients_required):

    if gradients_required:
        grad_context = torch.no_grad()
    else:
        grad_context = nullcontext()


    def update(engine: Engine, frame):
        with grad_context:

            engine.fire_event(MDMainEvents.INITIAL_INTEGRATE)
            engine.fire_event(MDMainEvents.POST_INTEGRATE)
            engine.fire_event(MDMainEvents.PRE_NEIGHBOR)
            engine.fire_event(MDMainEvents.NEIGHBOR)
            engine.fire_event(MDMainEvents.POST_NEIGHBOR)
            engine.fire_event(MDMainEvents.PRE_FORCE)
            engine.fire_event(MDMainEvents.FORCE)
            engine.fire_event(MDMainEvents.POST_FORCE)
            engine.fire_event(MDMainEvents.FINAL_INTEGRATE)
            engine.fire_event(MDMainEvents.END_STEP)

        return frame


    return update