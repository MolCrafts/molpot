from abc import ABC, abstractmethod
from typing import Any


class MolPotApp(ABC):
    def __init__(self): ...


class ConfigProcessor(ABC):

    def __init__(self, config: dict):
        self.config = self.check_header(config)

    @abstractmethod
    def check_header(self, config: dict) -> dict:
        return config

    @abstractmethod
    def process(self) -> dict[str, Any]: ...
