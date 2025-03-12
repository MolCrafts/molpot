from typing import Sequence

from ignite.engine import Engine, EventEnum, Events
from torch import nn


class MDMainEvents(EventEnum):
    """
    timestep execution events. The protocol is adapted from: https://docs.lammps.org/Developer_flow.html
    """
    STARTED = Events.STARTED.value

    START_STEP = "start_step"
    INITIAL_INTEGRATE = "initial_integrate"
    POST_INTEGRATE = "post_integrate"
    PRE_NEIGHBOR = "pre_neighbor"
    NEIGHBOR = "neighbor"
    POST_NEIGHBOR = "post_neighbor"
    PRE_FORCE = "pre_force"
    FORCE = "force"
    POST_FORCE = "post_force"
    FINAL_INTEGRATE = "final_integrate"
    END_STEP = "end_step"

    COMPLETED = Events.COMPLETED.value


class MDEvent(nn.modules.lazy.LazyModuleMixin, nn.Module):

    events: set[EventEnum]
    priorities: Sequence[int]

    def __init__(self, events: set[EventEnum], priorities: Sequence[int]):
        super().__init__()
        self.events = events
        self.priorities = priorities
        assert len(events) == len(priorities), "events and priorities should have the same length"

    def on_started(self, engine: Engine):
        ...

    def on_completed(self, engine: Engine):
        ...


class Potential(MDEvent):

    def __init__(self, potential: nn.Module):
        super().__init__(
            {MDMainEvents.FORCE},
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
    