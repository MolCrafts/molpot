from typing import Sequence
from torch.nn import Module
from .events import MDMainEvents, EventEnum, Events
from ignite.engine import Engine

class MDHandler:

    def __init__(self, events: set[EventEnum], priorities: Sequence[int]):
        self.events = events
        self.priorities = priorities
        assert len(events) == len(priorities), "events and priorities should have the same length"

    def attach(self, engine: Engine):

        for event in self.events:
            engine.add_event_handler(event, getattr(self, "on_" + event.value))


class Potential(MDHandler):

    def __init__(self, potential: Module):
        self.potential = potential
        super().__init__(
            {MDMainEvents.FORCE},
            (0, ),
        )

    def on_pre_force(self, engine: Engine):
        return engine
    
    def on_force(self, engine: Engine):
        frame = engine.state.frame
        self.potential(frame)
        return engine
    
    def on_post_force(self, engine: Engine):
        return engine
    