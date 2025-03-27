from typing import Callable
from molpot.engine.md.event import MDMainEvents
from molpot.engine.md.handler import MDHandler
from torch.nn import Module
from molpot import alias


class Thermo(MDHandler):

    kernels = {
        "step": lambda state: state.iterations - 1,
        "etotal": lambda state: state.frame["predicts", "energy"].item(),
        "volume": lambda state: state.frame[alias.cell][:, 0]
        @ state.frame[alias.cell][:, 1].cross(state.frame[alias.cell][:, 2]),
        "ndensity": lambda state: state.frame[alias.R].shape[0]
        / state.thermo["volume"],
    }

    def __init__(self, every: int, *keywords: str):
        super().__init__("thermo", {MDMainEvents.END_STEP(every=every)}, (0,))
        self.keywords = keywords
        self._kernels = [Thermo.kernels[key] for key in keywords]

    def on_end_step(self, engine):
        state = engine.state
        state.thermo.update(
            {key: kernel(state) for key, kernel in zip(self.keywords, self._kernels)}
        )
        return engine

    @classmethod
    def register(cls, name, kernel):
        cls.kernels[name] = kernel


class ThermoOutput(MDHandler):

    def __init__(self, every: int, output_handler: Callable = print):
        super().__init__("thermo_output", {MDMainEvents.END_STEP(every=every)}, (0,))
        self.output_handler = output_handler

    def on_end_step(self, engine):
        state = engine.state
        self.output_handler(state.thermo)
        return engine
