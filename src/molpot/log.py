import logging
import sys
from pathlib import Path
from ._config import Config

logger_initialized = {}

def setup_logger(name: str = None, output_dir: str|Path = None, rank: int = 0,
                 log_level: int|None = None) -> logging.Logger:
    """Initialize the logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only the logger of the master
    process is added console handler. If ``output_dir`` is specified, all loggers
    will be added file handler.

    Args:
        name (str): Logger name. Defaults to None to setup root logger.
        output_dir (str): The directory to save log.
        rank (int): Process rank in the distributed training. Defaults to 0.
        log_level (int): Verbosity level of the logger. Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: A initialized logger.
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    log_level = log_level or Config.log_level
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    formatter = logging.Formatter("[%(asctime)s %(name)s %(levelname)s]: %(message)s",
                                  datefmt="%m/%d %H:%M:%S")
    
    # create console handler for master process
    if rank == 0:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(output_dir / f"log_rank{rank}.txt")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger