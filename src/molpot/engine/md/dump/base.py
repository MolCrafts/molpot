from molpot.engine.md.handler import MDHandler
from molpot.engine.md.event import MDMainEvents

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

    def __init__(self, name, every=1, before=None, after=None):

        super().__init__(name, (MDMainEvents.END_STEP(every=every, before=before, after=after), ), (5,))
        self._frames = []

    @property
    def frames(self):
        return self._frames

    def on_end_step(self, engine):
        self._frames.append(engine.state.frame)
        return engine
