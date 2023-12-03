from .readers import *
from .datasets import *
from .dataloaders import *
from torchdata.datapipes.iter import *


def get_data_loader(config: dict) -> DataLoader2:
    data_loader_name = config["type"]
    data_loader_class = globals()[data_loader_name]
    data_loader_ins = data_loader_class(**config["args"])

    return data_loader_ins
