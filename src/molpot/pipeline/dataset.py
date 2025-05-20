from pathlib import Path
from tempfile import TemporaryDirectory
from molpot import NameSpace
from typing import Any, Sequence
from .process.base import ProcessManager


import tempfile
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

       
        self.processes = ProcessManager()
        #secure downloading process from configuration 
        processes_cfg = getattr(config, "processes", None)
        if processes_cfg is None:
            raise logger.info(f"No `processes` section in config, skipping processing for dataset `{name}`")
            processes_cfg = {}
        elif not isinstance(processes_cfg, dict):
            raise TypeError(f"Expected `config.processes` to be a dict, got {type(processes_cfg)}")
        #we take only section which is about our dataset
        proc_params = processes_cfg.get(name, {})
        if not isinstance(proc_params, dict):
            raise TypeError(f"`config.processes[{name}]` should be a dict of process parameters")
        #iterate over process parameters and add them to process manager
        for proc_name, params in proc_params.items():
            try:
                proc_cls = getattr(mpot.process, proc_name)
            except AttributeError:
                raise ValueError(f"Unknown process `{proc_name}` specified for dataset `{name}`")
            if not isinstance(params, dict):
                raise TypeError(f"Parameters for process `{proc_name}` must be a dict, got {type(params)}")
            self.processes.add(proc_cls(**params))
        
        self.frames: list[mpot.Frame] = []

    def prepare(self) -> int: 
        """
        Preparing frame list
        - download
        - load file by exact loader
        - fill self.frames
        returns quantity of examples loaded
        """
        self.download()
        loader_cls = DATASET_LOADER_MAP[self.name].get()
        if loader_cls is None:
            raise ValueError(f"No loader found for dataset {self.name}")
        loader = loader_cls(self.save_dir)
        self.frames = loader.load()
        logger.info(f"[prepare] {self.name} loaded {len(self.frames)} frames")
        return len(self.frames)
       

    def download(self): 
        #Download raw data and saves in self.save_dir if file already exists does nothing
        url = config.urls.get(self.name)
        if not url:
            logger.info(f"No URL found for dataset {self.name}")
            return 

        target = self.save_dir / Path(url).name
        if target.exists():
            logger.info(f"[download] {target.name} already exists")
            return
        
        logger.info(f"[download] {target.name} from {url} -> {target}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"[download] {target} saved")
        


    
    def get_frame(self, idx: int) -> mpot.Frame: 

        if not self.frames:
            raise RuntimeError(f"Data set has not been prepared yet")
        return self.frames[idx]
    

    def update(self, state: dict):


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

    def __getitem__(self, idx):
        return self.processes.process_one(self.get_frame(idx))
