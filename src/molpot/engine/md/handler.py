from typing import Sequence

from ignite.engine import Engine, EventEnum
from torch import nn
from .event import MDEvents
from ..base import Handler


class MDHandler(Handler):

    def __init__(self, name: str, events: set[EventEnum], priorities: Sequence[int]):
        super().__init__(name, events, priorities)

    def get_event_handler(self, event: EventEnum):
        return getattr(self, f"on_{event.name.lower()}", None)

    def on_started(self, engine: Engine):
        ...

    def on_completed(self, engine: Engine):
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"


class Potential(MDHandler):

    def __init__(self, potential: nn.Module):
        super().__init__(
            "potential",
            {MDEvents.FORCE},
            (0, ),
        )
        self.potential = potential

    def on_pre_force(self, engine: Engine):
        return engine
    
    def on_force(self, engine: Engine):
        frame = engine.state.frame
        self.potential(frame)
        return engine
    
    def on_post_force(self, engine: Engine):
        return engine
    