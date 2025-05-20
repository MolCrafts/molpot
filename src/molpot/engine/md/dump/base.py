from molpot.engine.md.handler import MDHandler
from molpot.engine.md.event import MDEvents


class Dump(MDHandler):

    _dump_instance = {}

    def __new__(cls, name, *args, **kwargs):
        ins = super().__new__(cls)
        cls._dump_instance[name] = ins
        return ins

    def __init__(self, name, event, priority):
        super().__init__(name, event, priority)

    @classmethod
    def get_dump(cls, name):
        return cls._dump_instance[name]


class DumpFrame(Dump):

    def __init__(self, name, every=1, before=None, after=None, label_thermo=False):

        super().__init__(
            name, (MDEvents.END_STEP(every=every, before=before, after=after),), (5,)
        )
        self.label_thermo = label_thermo
        self._frames = []

    @property
    def frames(self):
        return self._frames

    def on_end_step(self, engine):
        frame = engine.state.frame
        if self.label_thermo:
            frame["thermo"] = engine.state.thermo
        self._frames.append(engine.state.frame.copy())
        return engine
