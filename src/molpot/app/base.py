import json
import molpot as mpot
from pathlib import Path
import typer

class App:

    def __init__(self):
        self.cli = typer.Typer()
        self.logger = mpot.get_logger(f"molpot:{self.__class__.__name__}")

    def load_config(self, path: Path, format='json'):
        match format:
            case 'json':
                self.config = json.loads(path.open().read_text())

        return self.config

    def run_cli(self):
        self.cli()