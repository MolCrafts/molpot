import os

import torch


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # want to run __init__ every time the class is called, add
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=Singleton):

    device: torch.device = torch.device("cpu")
    ftype: torch.dtype = torch.float32
    stype: torch.dtype = torch.int32
    itype: torch.dtype = torch.complex64

    def __init__(self):
        # 0 = all messages are logged (default behavior)
        # 1 = INFO messages are not printed
        # 2 = INFO and WARNING messages are not printed
        # 3 = INFO, WARNING, and ERROR messages are not printed
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    @classmethod
    def set_device(cls, device_info:dict):
        device_type = device_info["type"]
        if device_type == "cpu":
            device = torch.device("cpu")
        elif device_type == "gpu" or device_type == "cuda":
            n_gpu = torch.cuda.device_count()
            if n_gpu == 0:
                print("Warning: There\'s no GPU available on this machine,"
                    "training will be performed on CPU.")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:0")
        return device
    
    @classmethod
    def set_environ(cls, **kwargs):
        for k, v in kwargs.items():
            os.environ[k] = v
