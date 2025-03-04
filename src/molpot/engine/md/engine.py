from contextlib import nullcontext
from pathlib import Path

from molpot.engine.md.initializer import Zeros
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from torch.nn import Module

from ..base import MolpotEngine
from .events import MDMainEvents
from .handler import MDHandler, Potential


def _infinite_iterator(frame):
    while True:
        yield frame


def main_process(gradients_required):

    if gradients_required:
        grad_context = torch.no_grad()
    else:
        grad_context = nullcontext()

    def update(engine: Engine, frame):
        with grad_context:

            engine.fire_event(MDMainEvents.START_STEP)
            engine.state.start_step += 1
            engine.fire_event(MDMainEvents.INITIAL_INTEGRATE)
            engine.state.initial_integrate += 1
            engine.fire_event(MDMainEvents.POST_INTEGRATE)
            engine.state.post_integrate += 1
            engine.fire_event(MDMainEvents.PRE_NEIGHBOR)
            engine.state.pre_neighbor += 1
            engine.fire_event(MDMainEvents.NEIGHBOR)
            engine.state.neighbor += 1
            engine.fire_event(MDMainEvents.POST_NEIGHBOR)
            engine.state.post_neighbor += 1
            engine.fire_event(MDMainEvents.PRE_FORCE)
            engine.state.pre_force += 1
            engine.fire_event(MDMainEvents.FORCE)
            engine.state.force += 1
            engine.fire_event(MDMainEvents.POST_FORCE)
            engine.state.post_force += 1
            engine.fire_event(MDMainEvents.FINAL_INTEGRATE)
            engine.state.final_integrate += 1
            engine.fire_event(MDMainEvents.END_STEP)
            engine.state.end_step += 1

        return 0  # TODO: return status because state.output should be an int

    return update


class MoleculeDymanics(MolpotEngine):

    main: Engine

    def __init__(self, grad_required=False, work_dir=Path.cwd()):

        super().__init__(work_dir=work_dir)

        self.grad_required = grad_required

        main_engine = Engine(main_process(gradients_required=self.grad_required))

        main_engine.register_events(
            *MDMainEvents, event_to_attr={event: event.value for event in MDMainEvents}
        )

        self.add_engine("main", main_engine)


    def add_handler(self, handler: MDHandler):

        for event in handler.events:
            self.main.add_event_handler(event, getattr(handler, "on_" + event.value))

    def add_handlers(self, *handlers: MDHandler):

        events_handlers = {event: [] for event in MDMainEvents}
        for handler in handlers:
            for event, prio in zip(handler.events, handler.priorities):
                events_handlers[event].append((getattr(handler, "on_" + event.value), prio))

        for event, handlers in events_handlers.items():
            handlers.sort(key=lambda x: x[1])
            for handler, _ in handlers:
                self.main.add_event_handler(event, handler)

    def set_potential(self, potential: Module):

        self.add_handler(Potential(potential))

    def run(self, frame, steps):

        self.main.state.frame = self._init_frame(frame)
        self.main.run(_infinite_iterator(frame), max_epochs=steps, epoch_length=1)

    def _init_frame(self, frame):

        frame["atoms", "forces"] = torch.zeros_like(frame["atoms", "R"])
        frame["atoms", "momenta"] = torch.zeros_like(frame["atoms", "R"])

        return frame
