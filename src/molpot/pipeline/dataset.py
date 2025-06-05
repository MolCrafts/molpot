from pathlib import Path
from tempfile import TemporaryDirectory
import tempfile
from typing import Any, Sequence

import torch

import molpot as mpot
from molpot import NameSpace

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

        # if save_dir is None, use temporary directory
        if save_dir is None:
            save_dir = tempfile.mkdtemp(suffix=name)
        else:
            save_dir = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

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
