from molpot.md.handler import MDHandler
from molpot.md.event import MDMainEvents

class Dump(MDHandler):
    
    _dump_instance = {}

    def __init__(self, name, event, priority):
        super().__init__(name, event, priority)
        self._dump_instance[name] = self

    @staticmethod
    def get_dump(name):
        return Dump._dump_instance[name]


class DumpFrame(Dump):

    def __init__(self, name, every):

        super().__init__(name, MDMainEvents.END_STEP(every=every), (5,))
        self._frames = []

    @property
    def frames(self):
        return self._frames

    def on_end_step(self, engine):
        self._frames.append(engine.state.frame)
        return engine
