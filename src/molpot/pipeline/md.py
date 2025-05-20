import molpot as mpot
from ..engine.md import MoleculeDymanics


class MDDataset:  # Usuń dziedziczenie Dataset na razie
    
    DUMP_HANDLER_NAME = "dump_frame"

    def __init__(
        self,
        engine: MoleculeDymanics,
    ):
        from .dataset import Dataset  # Import lokalny, aby uniknąć cyklicznego importu
        self.engine = engine
        self.frames = []
        # Upewniamy się, że klasa faktycznie dziedziczy po Dataset
        if not issubclass(self.__class__, Dataset):
            # Podczas pierwszego ładowania dynamicznie modyfikujemy klasę
            self.__class__ = type('MDDataset', (self.__class__, Dataset), {})

    def reset_dump(self, every: int, after: int):
        dumper = mpot.md.DumpFrame(
            self.DUMP_HANDLER_NAME,
            every=every,
            after=after,
            label_thermo=True,
        )
        self.engine.add_handler(dumper)
        return dumper

    def prepare(self, init_frame, steps: int, every, after:int = 0):
        dumper = self.reset_dump(every, after)
        self.engine.run(
            init_frame,
            steps=steps,
        )
        self.frames = dumper.frames
        return dumper.frames

    def resume(self, steps: int):
        self.engine.run(
            self.frames[-1],
            steps=steps,
        )
        self.frames = self.engine.get_handler(self.DUMP_HANDLER_NAME).get_frames()
        return self.frames

    def __len__(self):
        return len(self.frames)
