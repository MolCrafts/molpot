from molpot.engine.md.events import MDMainEvents
from .handler import MDHandler

class ThermoOutput(MDHandler):

    def __init__(self, every: int, keywords: list[str]):
        super().__init__(
            {MDMainEvents.END_STEP(every=every)},
            (0,)
        )

        self.keywords = keywords

    def on_end_step(self, engine):
        thermo = engine.state.thermo
        print(thermo)