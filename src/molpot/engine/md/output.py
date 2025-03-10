from typing import Callable
from molpot.engine.md.events import MDMainEvents
from .handler import MDEvent
from ignite.engine import Events
from torch.nn import Module
from molpot import alias

class Thermo(Module):

    kernels = {
        "step": lambda state: state.iterations - 1,
        "etotal": lambda state: state.frame["predicts", "energy"],
        "volume": lambda state: state.frame[alias.cell][:, 0] @ state.frame[alias.cell][:, 1].cross(state.frame[alias.cell][:, 2]),
        "ndensity": lambda state: state.frame[alias.R].shape[0] / state.thermo["volume"]
    }

    def __init__(self, *keywords):
        self.keywords = keywords
        self._kernels = [
            self.kernels[key] for key in self.keywords
        ]

    def __call__(self, state):
        state.thermo.update({key: kernel(state) for key, kernel in zip(self.keywords, self._kernels)})
        return state.thermo


class ThermoOutput(MDEvent):

    def __init__(self, every: int, thermo: Thermo, *output_handlers: Callable):
        super().__init__(
            {MDMainEvents.END_STEP(every=every)},
            (0,)
        )

        self.thermo = thermo
        self.output_handlers = output_handlers

    def on_end_step(self, engine):
        thermo = self.thermo(engine.state)
        for oh in self.output_handlers:
            oh({thermo[key] for key in self.keywords})