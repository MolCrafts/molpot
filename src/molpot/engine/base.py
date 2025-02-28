from abc import ABC

from ignite.engine import Engine


class MolpotEngine(ABC):

    def __init__(self):

        self._engines: dict[str, Engine] = {}

    def add_engine(self, name: str, engine: Engine):

        self._engines[name] = engine
