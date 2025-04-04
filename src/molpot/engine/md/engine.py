from contextlib import nullcontext
import molpot as mpot
import torch
from ignite.engine import Engine, State
from torch.nn import Module

from ..base import MolpotEngine, HandlerManager
from .event import MDEvents
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

            engine.fire_event(MDEvents.START_STEP)
            engine.state.start_step += 1
            engine.fire_event(MDEvents.INITIAL_INTEGRATE)
            engine.state.initial_integrate += 1
            engine.fire_event(MDEvents.POST_INTEGRATE)
            engine.state.post_integrate += 1
            engine.fire_event(MDEvents.PRE_NEIGHBOR)
            engine.state.pre_neighbor += 1
            engine.fire_event(MDEvents.NEIGHBOR)
            engine.state.neighbor += 1
            engine.fire_event(MDEvents.POST_NEIGHBOR)
            engine.state.post_neighbor += 1
            engine.fire_event(MDEvents.PRE_FORCE)
            engine.state.pre_force += 1
            engine.fire_event(MDEvents.FORCE)
            engine.state.force += 1
            engine.fire_event(MDEvents.POST_FORCE)
            engine.state.post_force += 1
            engine.fire_event(MDEvents.FINAL_INTEGRATE)
            engine.state.final_integrate += 1
            engine.fire_event(MDEvents.END_STEP)
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



class MoleculeDymanics(MolpotEngine):

    main: Engine
    events = MDEvents

    def __init__(self, grad_required=False):

        super().__init__()

        self.grad_required = grad_required

        main_engine = Engine(main_process(gradients_required=self.grad_required))

        main_engine.register_events(
            *MDEvents, event_to_attr={event: event.value for event in MDEvents}
        )

        self.add_engine("main", main_engine)


    def set_potential(self, potential: Module):

        self.add_handler(Potential(potential))

    def run(self, frame, steps):

        self._handlers.register_to_engine(self.main)
        self.main.state.frame = self._init_frame(frame)
        self.main.state.thermo = self._init_thermo()
        self.main.run(_infinite_iterator(frame), max_epochs=steps, epoch_length=1)

    def _init_frame(self, frame):

        if "predicts" not in frame:
            frame["predicts"] = {}
        if ("predicts", "forces") not in frame:
            frame["predicts", "forces"] = torch.zeros_like(frame["atoms", "R"])
        if ("atoms", "velocity") not in frame:
            frame["atoms", "velocity"] = torch.zeros_like(frame["atoms", "R"])
        if ("atoms", "momenta") not in frame:
            frame["atoms", "momenta"] = torch.zeros_like(frame["atoms", "R"])

        return frame

    def _init_thermo(self):
        return {"T": 1.0}
