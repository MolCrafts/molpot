import logging
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os, sys
import torch
from collections import defaultdict
import numpy as np
import threading


def _setup_logger():
    logger = logging.getLogger("molpot")
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # logger.setLevel(logging.INFO)
    return logger

class Config:

    _instance = None
    _lock = threading.Lock()

    device: torch.device = torch.device("cpu")
    global_dtypes = {
        "float": torch.float32,
        "int": torch.int32,
    }
    ftype = global_dtypes["float"]
    itype = global_dtypes["int"]

    logger = _setup_logger()

    seed: int|None = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_dtype(cls, dtype_name):
        return cls.global_dtypes.get(dtype_name, None)

    @classmethod
    def set_device(cls, device_info: dict):
        device_type = device_info["type"]
        if device_type == "cpu":
            device = torch.device("cpu")
        elif device_type == "gpu" or device_type == "cuda":
            n_gpu = torch.cuda.device_count()
            if n_gpu == 0:
                cls.logger.warning("There's no GPU available on this machine, training will be performed on CPU.")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:0")
        return device

    @classmethod
    def set_environ(cls, **kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v

    @classmethod
    def get_environ(cls):
        env_info = {}
        env_info["sys.platform"] = sys.platform
        env_info["Python"] = sys.version.replace("\n", "")
        env_info["Numpy"] = np.__version__

        cuda_available = torch.cuda.is_available()
        env_info["CUDA available"] = cuda_available

        if cuda_available:
            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
            for name, device_ids in devices.items():
                env_info["GPU " + ",".join(device_ids)] = name

        env_info["PyTorch"] = torch.__version__

        return env_info

    @classmethod
    def set_seed(cls, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def set_log_level(cls, level: int|str):
        _mapping = {
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        if isinstance(level, str):
            level = _mapping.get(level.upper())
            if level is None:
                raise ValueError(f"Invalid log level: {level}")
        cls.logger.setLevel(level)

    @classmethod
    def get_generator(cls):
        gen = torch.Generator(device=cls.device)
        if cls.seed is not None:
            gen = gen.manual_seed(cls.seed)
        return gen

