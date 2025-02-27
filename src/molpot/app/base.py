from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import inspect
from typer import Typer
import os

class ConfigParser:

    def __init__(self):
        self._config: dict[str, Any] = {}

    @property
    def config(self) -> dict[str, Any]:
        config = getattr(self, '_config', None)
        if config is None:
            raise AttributeError('Config is not set, capture config first')
        return config
    
    @config.setter
    def config(self, config: dict[str, Any]) -> None:
        self._config = config

from burr.core import Action


class MolpotApp(Action):

    name: str = None
    version: str = None

    def __init__(self) -> None:
        super().__init__()
        self.cli = Typer(chain=True)

    def with_config(self, *args, **kwargs):
        ...

class TrainPotentialApp(MolpotApp):
    ...