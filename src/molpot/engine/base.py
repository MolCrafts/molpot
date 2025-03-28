from abc import ABC
from pathlib import Path
from ignite.engine import Engine


class MolpotEngine(ABC):

    def __init__(self, work_dir: Path=Path.cwd()):

        self._engines: dict[str, Engine] = {}
        self._work_dir = Path(work_dir).absolute()
        self._work_dir.mkdir(parents=True, exist_ok=True)

    @property
    def work_dir(self) -> Path:
        
        return self._work_dir
    

    def get_absolute_path(self, path: Path|str) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.work_dir / path


    def add_engine(self, name: str, engine: Engine):

        self._engines[name] = engine
        setattr(self, name, engine)