from typing import Any
import inspect
from typer import Typer

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

    def capture_config(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # get args name and their values
        frame = inspect.currentframe().f_back
        args_name = inspect.getargvalues(frame).args[1:]
        config = dict(zip(args_name, args))
        config.update(kwargs)
        self.config = config

    def save_config(self, path: str, format: str = 'yaml') -> None:
        match format:
            case 'json':
                import json
                with open(path, 'w') as f:
                    json.dump(self.config, f)
            case _:
                raise ValueError(f'Unsupported format: {format}')

    def load_config(self, path: str, format: str = 'yaml') -> None:
        ...


class App:

    name: str = 'MolPotApp'
    version: str = '0.1.0'
    cli = Typer()

    def __init__(self, *args: Any, **configs: Any) -> None:
        self.config = ConfigParser()
        self.config.capture_config(*args, **configs)

class TrainPotentialApp(App):

    def __init__(self, *args: tuple, **configs: dict) -> None:
        super().__init__(*args, **configs)
        self._model = None
        self.cli.command()(self)
        self.cli.command()(self.train)


    def train(self):
        ...

class PiNetApp(TrainPotentialApp):

    def __init__(self, a:int, b:str):
        super().__init__(a=a, b=b)
    