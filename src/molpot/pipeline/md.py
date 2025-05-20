import molpot as mpot
from ..engine.md import MoleculeDymanics


class MDDataset:  # Remove Dataset inheritance for now
    
    DUMP_HANDLER_NAME = "dump_frame"

    def __init__(
        self,
        engine: MoleculeDymanics,
    ):
        from .dataset import Dataset  # Local import to avoid circular import
        self.engine = engine
        self.frames = []
        # Make sure the class actually inherits from Dataset
        if not issubclass(self.__class__, Dataset):
            # During the first load, dynamically modify the class
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
