from contextlib import nullcontext
import molpot as mpot
import torch
from ignite.engine import Engine, State
from torch.nn import Module

from ..base import MolpotEngine
from .event import MDMainEvents
from .handler import MDHandler, Potential

logger = mpot.get_logger("molpot.md")


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


class MDState(State):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._integrator = None
        self._frame: mpot.Frame = None

    @property
    def integrator(self):
        return self._integrator

    @integrator.setter
    def integrator(self, integrator):
        if self._integrator is not None:
            logger.warning(f"Overwriting existing integrator {self._integrator}.")
        self._integrator = integrator

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame


class MDEngine(Engine):

    def __init__(self, process_function):

        super().__init__(process_function)

        self.state = MDState()


class HandlerManager:

    def __init__(self):

        self._handlers: dict[str, MDHandler] = {}
        self._registered: set = set()

    @property
    def handlers(self):
        return self._handlers

    def add_handler(self, handler: MDHandler):
        self._handlers[handler.name] = handler

    def get_handlers_by_event(self, sort=True) -> dict[MDMainEvents, list[MDHandler]]:
        events_handlers = {event: [] for event in MDMainEvents}
        for handler in self._handlers.values():
            for event in handler.events:
                events_handlers[event].append(handler)
        if sort:
            for event, handlers in events_handlers.items():
                handlers.sort(key=lambda x: x.priorities)
        return events_handlers

    def register_to_engine(self, engine: Engine):

        events_handlers = self.get_handlers_by_event()
        to_be_registered = set()
        for event_enum, handlers in events_handlers.items():
            for handler in handlers:
                if handler.name in self._registered:
                    continue
                handler_event = next(
                    (e for e in handler.events if e == event_enum), None
                )  # event_enum is defined in MDMainEvents
                # it's without every info
                engine.add_event_handler(
                    handler_event, handler.get_event_handler(event_enum)
                )
                to_be_registered.add(handler.name)
        self._registered.update(to_be_registered)


class MoleculeDymanics(MolpotEngine):

    main: Engine

    def __init__(self, grad_required=False):

        super().__init__()

        self.grad_required = grad_required

        main_engine = Engine(main_process(gradients_required=self.grad_required))

        main_engine.register_events(
            *MDMainEvents, event_to_attr={event: event.value for event in MDMainEvents}
        )

        self.add_engine("main", main_engine)

        self._handlers = HandlerManager()

    def add_handler(self, handler: MDHandler):

        self._handlers.add_handler(handler)

    def add_handlers(self, *handlers: MDHandler):

        for handler in handlers:
            self.add_handler(handler)

    def get_handler(self, name: str) -> MDHandler:

        return self._handlers.handlers[name]

    def set_potential(self, potential: Module):

        self.add_handler(Potential(potential))

    def run(self, frame, steps):

        self._handlers.register_to_engine(self.main)
        self.main.state.frame = self._init_frame(frame)
        self.main.state.thermo = self._init_thermo()
        self.main.run(_infinite_iterator(frame), max_epochs=steps, epoch_length=1)

    def _init_frame(self, frame):

        frame["predicts"] = {}
        frame["predicts", "forces"] = torch.zeros_like(frame["atoms", "R"])
        frame["predicts", "momenta"] = torch.zeros_like(frame["atoms", "R"])

        return frame

    def _init_thermo(self):
        return {}
