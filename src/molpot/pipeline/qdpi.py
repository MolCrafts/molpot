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
from .dataset import MapStyleDataset
import h5py

logger = mpot.get_logger("molpot.dataset")
config = mpot.get_config()

class QDpi(MapStyleDataset):

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
        super().__init__("QDpi", save_dir=save_dir, device=device)
        self.subset = subset


    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]
    
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
        logger.info("prepaering QDpi dataset...")

        ds = self.get_subset_data()

        for category, subds in ds.items():
            for key, url in subds.items():
                self._download(url, key)

    def _download(self, url: str, key: str):
        gitlab_root = "https://gitlab.com/RutgersLBSR/QDpiDataset/-/raw/main/data/"
        qdpi_hdf5 = f"{self.save_dir}/{key}.hdf5"
        with requests.get(gitlab_root+url, stream=True) as response:
            response.raise_for_status()
            with open(qdpi_hdf5, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        with h5py.File(qdpi_hdf5, 'r') as f:
            
            for name, mol in f.items():
                pbc = not bool(mol["nopbc"])
                coord = torch.tensor(mol["set.000"]["coord.npy"]).reshape(-1, 3)
                energy = torch.tensor(mol["set.000"]["energy.npy"])  # (1, 1)
                force = torch.tensor(mol["set.000"]["force.npy"]).reshape(-1, 3)
                net_charge = torch.tensor(mol["set.000"]["net_charge.npy"])  # (1, 1)
                type_raw = torch.tensor(mol["type.raw"])
                type_map = torch.tensor(mol["type.map"])
                type_name = mpot.Element.get_atomic_number(type_map[type_raw])

                frame = mpot.Frame()
                frame[alias.R] = coord
                frame[alias.F] = force
                frame[alias.E] = energy
                frame[alias.Q] = net_charge
                frame[alias.Z] = type_name

                self._frames.append(frame)

        return self._frames