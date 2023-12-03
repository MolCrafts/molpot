import logging
import logging.config

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
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "datetime",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "info_file_handler"]},
}


class LogAdapter:
    def __init__(
        self, name: str, save_dir, config=DEFAULT_CONFIG, default_level=logging.INFO
    ):
        self.logger = logging.getLogger(name)
        logging.config.dictConfig(config)
        logging.basicConfig(level=default_level)

    def info(self, msg):
        self.logger.info(msg)
