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

def create_dataloader(dp: IterDataPipe, batch_size: int, num_workers: int = 0):
    if num_workers:
        rs = MultiProcessingReadingService(num_workers=4)
        return DataLoader2(dp, reading_service=rs)
    else:
        return DataLoader2(dp)