import logging
import logging.config
from pathlib import Path
from molpot.trainer.utils import read_json

DEFAULT_CONFIG = 
{
    "version": 1, 
    "disable_existing_loggers": False, 
    "formatters": {
        "simple": {"format": "%(message)s"}, 
        "datetime": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    }, 
    "handlers": {
        "console": {
            "class": "logging.StreamHandler", 
            "level": "DEBUG", 
            "formatter": "simple", 
            "stream": "ext://sys.stdout"
            }, 
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler", 
            "level": "INFO", 
            "formatter": "datetime", 
            "filename": "info.log", 
            "maxBytes": 10485760, 
            "backupCount": 20, "encoding": "utf8"
        }
    }, 
    "root": {
        "level": "INFO", 
        "handlers": [
            "console", 
            "info_file_handler"
        ]
    }
}

def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)