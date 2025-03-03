from typing import Sequence
import torch
from torch.nn import Module
from .events import MDMainEvents, EventEnum
from ignite.engine import Engine

class MDHandler:

    def __init__(self, events: set[EventEnum], priorities: Sequence[int]):
        self.events = events
        self.priorities = priorities
        for event in self.events:
            # add a dummy handler for each event
            setattr(self, "on_" + event.value, lambda
                engine: engine
            )

    def attach(self, engine: Engine):

        for event in self.events:
            engine.add_event_handler(event, getattr(self, "on_" + event.value))


class Potential(MDHandler):

    def __init__(self, potential: Module):
        self.potential = potential
        super().__init__(
            {MDMainEvents.PRE_FORCE, MDMainEvents.FORCE, MDMainEvents.POST_FORCE},
            (0, 0, 0),
        )

    def on_force(self, engine: Engine):
        frame = engine.state.frame
        self.potential(frame)
        return engine