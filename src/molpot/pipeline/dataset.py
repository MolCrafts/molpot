import io
import tarfile
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import requests
import torch
from torch.nn import Module, Sequential
from tqdm import tqdm

import molpot as mpot
from molpot import Config, NameSpace, alias

from .process.base import ProcessManager

logger = mpot.get_logger("molpot.dataset")
config = mpot.get_config()


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name: str,
        save_dir: Path | None = None,
        device: str = "cpu"
    ):
        self.labels: NameSpace = NameSpace(name)
        self.device = device

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.exists():  # create save_dir
                self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        self.processes = ProcessManager()

    def prepare(self): ...

    def download(self): ...

    def update(self, state): ...

    def get_frame(self, idx: int) -> mpot.Frame: ...
        

class IterStyleDataset(Dataset):

    def __init__(self, frames: Sequence[mpot.Frame]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int) -> Any:
        return self.frames[i]


class MapStyleDataset(Dataset):

    def __getitem__(self, idx):
        frame = self.get_frame(idx)
        return self.processes.process_one(frame)