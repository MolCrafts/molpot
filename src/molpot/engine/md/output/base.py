from molpot.engine.md.event import MDEvents
from molpot.engine.md.handler import MDHandler
from molpot import alias

from rich.live import Live
from rich.table import Table



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
        super().__init__("thermo", {MDEvents.END_STEP(every=every)}, (0,))
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

    def __init__(self, every: int, *keys: str):
        super().__init__("thermo_output", {MDEvents.END_STEP(every=every)}, (0, ))

        self.table = Table()
        for key in keys:
            self.table.add_column(key, justify="right", style="cyan", no_wrap=True)
        self.live = Live(self.table, auto_refresh=False)

    def on_end_step(self, engine):
        state = engine.state
        self.table.add_row(
            *[f"{state.thermo[key.header]:.3f}" for key in self.table.columns],
        )
        self.live.update(
            self.table, refresh=True
        )
        return engine
    
