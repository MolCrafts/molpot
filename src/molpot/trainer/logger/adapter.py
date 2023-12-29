import logging
import logging.config
from pathlib import Path

__all__ = ["LogAdapter"]

DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(message)s"},
        "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "info.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]},
}


class LogAdapter:
    def __init__(
        self, name: str, keys=[], format="%(message)s", is_echo=True, save_dir=None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if save_dir:
            self.save_dir = Path(save_dir)
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)
            fileHandler = logging.handlers.RotatingFileHandler(self.save_dir / f"{name}.log")
            fileHandler.setFormatter(logging.Formatter(format))
            self.logger.addHandler(fileHandler)

        else:
            self.save_dir = None

        if is_echo:
            consoleHandler = logging.StreamHandler()
            consoleHandler.setFormatter(logging.Formatter(format))
            self.logger.addHandler(consoleHandler)
            row = [f"{key:<10}" for key in keys]
            msg = f"{'nstep':<10}" + f" ".join(row)
            self.logger.info(msg)
        self.keys = keys

    def __call__(self, nstep, output, data):
        row = [str(nstep)] + [f"{float(output[key]):<10.2f}" for key in self.keys]
        msg = f" ".join(row)
        self.logger.info(msg)
