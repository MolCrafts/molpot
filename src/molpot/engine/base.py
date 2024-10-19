from abc import ABC

from ignite.engine import Engine
from typing import Callable


class MolpotEngine(ABC):

    def __init__(self):

        self._engines: dict[str, Engine] = {}

    def add_engine(self, name: str, engine: Engine):

        self._engines[name] = engine

    def add_event(self, engine_name: str, event_name, handler: Callable):
        self._engines[engine_name].add_event_handler(event_name, handler)
