import typer
from pathlib import Path
import json
from typing_extensions import Annotated
import os
from .utils import SingletonCLI


class ConfigCLI(SingletonCLI):

    def __init__(self):
        super().__init__()
        self.load_config()

        self.cli.command()(self.set)
        self.cli.command()(self.get)

    def load_config(self):

        CONFIG_PATH = os.getenv("MOLCRAFTS_CONFIG_PATH")
        if CONFIG_PATH:
            CONFIG_PATH = Path(CONFIG_PATH)
        else:
            CONFIG_PATH = Path.home() / ".molcrafts_config.json"

        self.configs = {
            "app_dir": str(Path.home() / ".molpot_app"),
        }

        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r") as f:
                self.configs = json.load(f)["molpot"]

        self.CONFIG_PATH = CONFIG_PATH


    def set(self, key: str, value: str):
        """Set a config value"""
        self.configs[key] = value
        with open(self.CONFIG_PATH, "w") as f:
            json.dump({"molpot": self.configs}, f)

    def get(self, key: str):
        """Get a config value"""
        typer.echo(self.configs.get(key))
