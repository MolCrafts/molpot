import logging
import os, sys
import torch
from collections import defaultdict
import numpy as np
import threading
from .logging import get_logger

logger = get_logger()

class Config:

    _instance = {}
    _lock = threading.Lock()

    _device: torch.device = torch.device("cpu")
    global_dtypes = {
        "float": torch.float32,
        "int": torch.int32,
    }
    ftype = global_dtypes["float"]
    itype = global_dtypes["int"]

    seed: int|None = None

    def __new__(cls, name: str = "global"):
        with cls._lock:
            if name not in cls._instance:
                cls._instance[name] = super().__new__(cls)
        return cls._instance[name]
    
    def get_dtype(self, dtype_name):
        return self.global_dtypes.get(dtype_name, None)

    def set_device(self, device: str|torch.device) -> torch.device:
        if isinstance(device, torch.device):
            self._device = device
        elif isinstance(device, str):
            if device == "cpu":
                device = torch.device("cpu")
            elif device == "gpu" or device == "cuda":
                n_gpu = torch.cuda.device_count()
                if n_gpu == 0:
                    logger.warning("There's no GPU available on this machine, training will be performed on CPU.")
                    device = torch.device("cpu")
                else:
                    device = torch.device("cuda:0")
        self._device = device
        return device
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self.set_device(device)

    def set_environ(self, **kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v

    def get_environ(self):
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

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_log_level(self, level: int|str):
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
        logger.setLevel(level)

    def get_generator(self):
        gen = torch.Generator()  # caveat: mismatch device
        if self.seed is not None:
            gen = gen.manual_seed(self.seed)
        return gen

def get_config(name: str = "global") -> Config:
    return Config(name)