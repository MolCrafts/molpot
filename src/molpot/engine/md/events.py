from typing import Sequence

from ignite.engine import Engine, EventEnum
from torch.nn import Module


class MDMainEvents(EventEnum):
    """
    timestep execution events. The protocol is adapted from: https://docs.lammps.org/Developer_flow.html
    """
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



class MDEvent(Module):

    events: set[EventEnum]
    priorities: Sequence[int]

    def __init__(self, events: set[EventEnum], priorities: Sequence[int]):
        self.register_buffer("events", events, persistent=False)
        self.register_buffer("priorities", priorities, persistent=False)
        assert len(events) == len(priorities), "events and priorities should have the same length"

    def attach(self, engine: Engine):

        for event in self.events:
            engine.add_event_handler(event, getattr(self, "on_" + event.value))

    def on_engine_start(self, engine: Engine):
        ...

    def on_engine_end(self, engine: Engine):
        ...


class Potential(MDEvent):

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
    