from pathlib import Path
from tempfile import TemporaryDirectory
from molpot import NameSpace
from typing import Any, Sequence
from .process.base import ProcessManager, Process

import logging
import json
import torch
import requests
import molpot as mpot

logger = mpot.get_logger("molpot.dataset")
config = mpot.get_config()

from .md import MDDataset
from .qm9 import QM9
from .rmd17 import rMD17
from .qdpi import QDpi

DATASET_LOADER_MAP = {
    "md": MDDataset,
    "qm9": QM9,
    "rmd17": rMD17,
    "qdpi": QDpi,
}


def get_logger(name: str):
    return logging.getLogger(name)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, name: str, save_dir: Path | None = None, device: str = "cpu"):
        super().__init__()
        self.name = name
        self.labels = NameSpace(name)
        self.device = device

        if save_dir is None:
            self._tmpdir = TemporaryDirectory(suffix=f"_{name}")
            self.save_dir = Path(self._tmpdir.name)
        else:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Start with an empty ProcessManager; add steps explicitly via add_process()
        self.processes = ProcessManager()
        self.frames: list[mpot.Frame] = []

    def add_process(self, proc: Process) -> None:
        """
        Attach one processing step (e.g. AtomicDescriptor, CalcNeighborList) to this dataset.
        Example:
            ds = QM9(...)
            ds.add_process(AtomicDescriptor(cutoff=5.0))
            ds.add_process(CalcNeighborList(r=6.0))
        """
        self.processes.append(proc)

    def prepare(self) -> int:
        """
        Prepare frame list by:
         1. downloading data
         2. loading with the appropriate loader
         3. filling self.frames
        Returns the number of frames loaded.
        """
        self.download()

        loader_cls = DATASET_LOADER_MAP.get(self.name)
        if loader_cls is None:
            raise ValueError(f"No loader registered for dataset '{self.name}'")

        loader = loader_cls(self.save_dir)
        self.frames = loader.load()

        logger.info(f"[prepare] {self.name} loaded {len(self.frames)} frames")
        return len(self.frames)

    def download(self):
        """
        Download raw data into save_dir. If the file already exists, do nothing.
        """
        url = config.urls.get(self.name)
        if not url:
            logger.info(f"No URL found for dataset {self.name}")
            return

        target = self.save_dir / Path(url).name
        if target.exists():
            logger.info(f"[download] {target.name} already exists")
            return

        logger.info(f"[download] downloading {target.name} from {url}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"[download] {target} saved")

    def get_frame(self, idx: int) -> mpot.Frame:
        if not self.frames:
            raise RuntimeError("Dataset.prepare() must be called before get_frame()")
        return self.frames[idx]

    def update(self, state: dict):
        """
        Save a state dictionary to state.json in save_dir.
        """
        state_file = self.save_dir / "state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        logger.debug(f"[update] {state_file} updated")

    def __len__(self) -> int:
        return len(self.frames)


class IterStyleDataset(Dataset):
    def __init__(self, frames: Sequence[mpot.Frame]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int) -> Any:
        return self.frames[i]


class MapStyleDataset(Dataset):
    def __init__(self, name, save_dir=None, device="cpu"):
        super().__init__(name, save_dir, device)
        self.prepare()

    def __getitem__(self, idx: int) -> Any:
        return self.processes.process_one(self.get_frame(idx))
