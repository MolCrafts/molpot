from functools import partial
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from typing import Optional, Any
from pathlib import Path
import tempfile
import time
import json
import logging
from urllib.request import urlretrieve
import numpy as np
import re
import tarfile
import molpot as mpot
from molpot import alias
import molpy as mp

__all__ = ['DataLoader', 'create_dataloader']

class DataLoader(DataLoader2):
    pass

def create_dataloader(datapipe: IterDataPipe, num_workers=Optional[int]=None) -> DataLoader:

    if num_workers:
        rs = MultiProcessingReadingService(num_workers)
    else:
        rs = None
    make_dataloader = partial(DataLoader, reading_service=rs)
    return make_dataloader(datapipe)