from functools import partial
import torch
from torchdata.datapipes.iter import IterDataPipe, FileLister, IterableWrapper
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

__all__ = ["Pipline", "QM9"]


log = logging.getLogger(__name__)


def endswith(x: str, suffix):
    return x.endswith(suffix)


class DataSet:

    """
    Base class for all datasets. It includes 5 processes:
        * Download / tokenize / process.
        * Clean and (maybe) save to disk.
        * Load inside Dataset.
        * Apply transforms (rotate, tokenize, etc…).
        * Wrap inside a DataLoader.
    """

    def __init__(
        self,
        name,
        data_dir: Optional[Path | str],
    ):
        super().__init__()
        self.name = name
        if data_dir is None:
            self.data_dir = Path(tempfile.mkdtemp())
        else:
            self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def update_meta(self, data: Optional[dict[str, Any]] = None):
        _data = {
            "name": self.name,
            "update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        _data.update(data or {})
        with open(self.data_dir / Path("meta.json"), "w") as f:
            json.dump(_data, f)
        self.is_prepared = True

    def set_keywords(self, keywords: dict[str, str]):
        self.keywords = keywords

    def prepare(self):
        raise NotImplementedError

    def fetch(self, url, filename, dir: Optional[Path | str] = None) -> Path:
        if dir is None:
            dir = self.data_dir

        fpath = Path(dir) / Path(filename)
        if not fpath.exists():
            log.info(f"downloading from {url}... to {fpath}")
            urlretrieve(url, fpath)
        else:
            log.info(f"{fpath} already exists.")
        return fpath


class QM9(DataSet):
    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("QM9", data_dir)
        self.remove_uncharacterized = remove_uncharacterized
        self.keywords = mpot.Keywords("QM9")
        self.keywords.set("A", "rotational_constant_A", "GHz")
        self.keywords.set("B", "rotational_constant_B", "GHz")
        self.keywords.set("C", "rotational_constant_C", "GHz")
        self.keywords.set("mu", "dipole_moment", "Debye")
        self.keywords.set("alpha", "isotropic_polarizability", "a0 a0 a0")
        self.keywords.set("homo", "homo", "Ha")
        self.keywords.set("lumo", "lumo", "Ha")
        self.keywords.set("gap", "gap", "Ha")
        self.keywords.set("r2", "electronic_spatial_extent", "a0 a0")
        self.keywords.set("zpve", "zpve", "Ha")
        self.keywords.set("U0", "energy_U0", "")
        self.keywords.set("U", "energy_U", "Ha")
        self.keywords.set("H", "enthalpy_H", "Ha")
        self.keywords.set("G", "free_energy", "Ha")
        self.keywords.set("Cv", "heat_capacity", "cal/mol/K")

    def prepare(self) -> DataLoader2:
        atomrefs = self._download_atomrefs()
        if self.remove_uncharacterized:
            uncharacterized = self._download_uncharacterized()
        else:
            uncharacterized = None
        ordered_files = self._download_data()

        irange = np.arange(len(ordered_files), dtype=int)
        if uncharacterized is not None:
            irange = np.setdiff1d(
                irange, np.array(uncharacterized, dtype=int) - 1
            )

        dp = (
            IterableWrapper((map(str, np.array(ordered_files)[irange])))
            # can not use lambda due to it is not pickleable
            .filter(filter_fn=partial(endswith, suffix=".xyz"))
            .open_files(mode="rt")
            .read_xyz(keywords=self.keywords).in_memory_cache()
        )

        rs = MultiProcessingReadingService(num_workers=1)
        dl = DataLoader2(dp, reading_service=rs)
        return dl

    def _download_atomrefs(self):
        url = "https://ndownloader.figshare.com/files/3195395"
        filename = "atomref.txt"
        atomrefs_path = self.fetch(url, filename)
        props = [
            self.keywords.zpve,
            self.keywords.U0,
            self.keywords.U,
            self.keywords.H,
            self.keywords.G,
            self.keywords.Cv,
        ]
        atref = {p: np.zeros((100,)) for p in props}
        with open(atomrefs_path) as f:
            lines = f.readlines()
            for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
                for i, p in enumerate(props):
                    atref[p][z] = float(l.split()[i + 1])
        atref = {k: v.tolist() for k, v in atref.items()}
        return atref

    def _download_uncharacterized(self):
        at_url = "https://ndownloader.figshare.com/files/3195404"
        path = self.fetch(at_url, "uncharacterized.txt")
        uncharacterized = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_data(self):
        url = "https://ndownloader.figshare.com/files/3195389"
        tar_path = self.fetch(url, "gdb9.tar.gz")
        raw_path = self.data_dir / Path("gdb9_xyz")

        if not raw_path.exists():
            log.info("Extracting files...")
            tar = tarfile.open(tar_path)
            tar.extractall(raw_path)
            tar.close()
            log.info("Done.")

        log.info("Parse xyz files...")
        ordered_files = sorted(
            raw_path.rglob("*.xyz"),
            key=lambda x: (int(re.sub(r"\D", "", str(x))), str(x)),
        )  # sort by index in filename
        return ordered_files
