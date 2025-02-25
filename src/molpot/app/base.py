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

class MolpotApp:

    name: str = 'MolPotApp'
    version: str = '0.1.0'

    def __init__(self, work_dir: Path = Path.cwd()) -> None:
        self.cli = Typer(chain=True)
        self.work_dir = work_dir
        self.config = ConfigParser()

        self._setup_workspace()

    def _setup_workspace(self) -> None:
        self.work_dir.mkdir(exist_ok=True)

class TrainPotentialApp(MolpotApp):

    def __init__(self, name: str, root: str, ):
        super().__init__(work_dir=Path(root)/name)
        os.chdir(self.work_dir)

    def run(self):
        ...

    def profile(self):
        ...
