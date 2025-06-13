import io
import tarfile
import time
from pathlib import Path

import numpy as np
import requests
import torch
from torch.nn import Sequential
from tqdm import tqdm

import molpot as mpot
from molpot import alias
from collections import defaultdict
import h5py

logger = mpot.get_logger("molpot.dataset")
config = mpot.get_config()

class QDpi:

    urls = {
        "charged": {
            "re_charged": "charged/re_charged.hdf5",
            "remd_charged": "charged/remd_charged.hdf5",
            "spice_charged": "charged/spice_charged.hdf5",
        },
        "neutral": {
            "ani": "neutral/ani.hdf5",
            "comp6": "neutral/comp6.hdf5",
            "freesolvmd": "neutral/freesolvmd.hdf5",
            "geom": "neutral/geom.hdf5",
            "re": "neutral/re.hdf5",
            "remd": "neutral/remd.hdf5",
            "spice": "neutral/spice.hdf5",
        }
    }

    def __init__(
        self,
        save_dir: Path | None = None,
        device: str = "cpu",
        subset: str|list[str] = "all",
    ):
        from .dataset import MapStyleDataset
        
        self.name = "QDpi"
        self.save_dir = save_dir
        self.device = device
        self.subset = subset
        
        if not issubclass(self.__class__, MapStyleDataset):
            self.__class__ = type('QDpi', (self.__class__, MapStyleDataset), {})
            MapStyleDataset.__init__(self, self.name, save_dir=self.save_dir, device=self.device)

    def __len__(self):
        return len(getattr(self, '_frames', []))

    def __getitem__(self, idx):
        return getattr(self, '_frames', [])[idx]
    
    def get_subset_data(self):
        if self.subset == "all":
            return self.urls

        ds = defaultdict(dict)
        missing = []

        def add_entry(key: str):
            for category in self.urls:
                if key in self.urls[category]:
                    ds[category][key] = self.urls[category][key]
                    return True
            return False

        if isinstance(self.subset, str):
            if not add_entry(self.subset):
                missing.append(self.subset)
        elif isinstance(self.subset, list):
            for key in self.subset:
                if not add_entry(key):
                    missing.append(key)

        if missing:
            print(f"[Warning] The following entries were not found: {missing}")

        return dict(ds)

    def prepare(self):
        self._frames = []  # Initialize frames list
        print(f"[DEBUG] QDpi.prepare() called, current frames: {len(self._frames)}")
        logger.info("preparing QDpi dataset...")

        ds = self.get_subset_data()

        for category, subds in ds.items():
            for key, url in subds.items():
                self._download(url, key)
        
        print(f"[DEBUG] QDpi.prepare() finished, frames: {len(self._frames)}")
        return self._frames

    def _download(self, url: str, key: str):
        gitlab_root = "https://gitlab.com/RutgersLBSR/QDpiDataset/-/raw/main/data/"
        qdpi_hdf5 = f"{self.save_dir}/{key}.hdf5"
        
        # Download if not exists
        if not Path(qdpi_hdf5).exists():
            logger.info(f"Downloading {key} dataset...")
            with requests.get(gitlab_root+url, stream=True) as response:
                response.raise_for_status()
                with open(qdpi_hdf5, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

        # Load data
        with h5py.File(qdpi_hdf5, 'r') as f:
            logger.info(f"Loading {len(f)} molecules from {key}")
            
            for name, mol in f.items():
                try:
                    # Basic data
                    pbc = not bool(mol["nopbc"])
                    coord = torch.tensor(np.array(mol["set.000"]["coord.npy"])).reshape(-1, 3)
                    energy = torch.tensor(np.array(mol["set.000"]["energy.npy"]))
                    force = torch.tensor(np.array(mol["set.000"]["force.npy"])).reshape(-1, 3)
                    
                    # Net charge might not exist for all molecules
                    if "net_charge.npy" in mol["set.000"]:
                        net_charge = torch.tensor(np.array(mol["set.000"]["net_charge.npy"]))
                    else:
                        net_charge = torch.tensor(0.0)  # Default to neutral
                    
                    # Type information
                    type_raw = torch.tensor(np.array(mol["type.raw"]))
                    type_map = [mol["type_map.raw"][i].decode() for i in range(len(mol["type_map.raw"]))]
                    type_name = torch.tensor([mpot.Element(type_map[i]).number for i in type_raw])

                    # Create frame
                    frame = mpot.Frame()
                    frame[alias.R] = coord
                    frame[alias.F] = force
                    frame[alias.E] = energy
                    frame[alias.Q] = net_charge
                    frame[alias.Z] = type_name

                    self._frames.append(frame)
                    
                except Exception as e:
                    logger.warning(f"Skipping molecule {name} due to error: {e}")
                    continue

        logger.info(f"Successfully loaded {len(self._frames)} frames")
        return self._frames 