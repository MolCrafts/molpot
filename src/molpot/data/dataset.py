# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-21
# version: 0.0.1

import json
import logging
import io
import re
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve

import molpy as mp
import molpot as mpot
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from molpot import keywords as kw
from molpot.configs.keywords import Keywords
from molpot.data.dataproxy import DataProxy

log = logging.getLogger(__name__)


class DataError(Exception):
    pass


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
        batch_size: int,
        n_train: Optional[int] = None,
        n_valid: Optional[int] = None,
        n_test: Optional[int] = None,
    ):
        self.name = name
        if data_dir is None:
            self.data_dir = tempfile.mkdtemp()
        else:
            self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test

        self.is_prepared = False

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
        if not Path(dir/filename).exists():
            log.info(f"downloading from {url}...")
            urlretrieve(url, fpath)
        else:
            log.info(f"{fpath} already exists.")
        return fpath

    def load_data(self, dir) -> DataProxy:
        raise NotImplementedError

    def load_dataproxy(self, datapath, format) -> DataProxy:
        raise NotImplementedError


class QM9DataSet(DataSet):
    def __init__(
        self,
        data_dir: Optional[Path | str],
        batch_size: int,
        n_train: Optional[int] = None,
        n_valid: Optional[int] = None,
        n_test: Optional[int] = None,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("QM9", data_dir, batch_size, n_train, n_valid, n_test)
        self.remove_uncharacterized = remove_uncharacterized
        self.keywords = Keywords("QM9")
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

        self.dataproxy: Optional[DataProxy] = None

    @property
    def properties(self) -> dict[str, str]:
        props = {}
        for kw in self.keywords:
            props[kw.keyword] = kw.unit
        return props

    def prepare(self) -> DataProxy:
        if not self.is_prepared:
            atomrefs = self._download_atomrefs()
            if self.remove_uncharacterized:
                uncharacterized = self._download_uncharacterized()
            else:
                uncharacterized = None
            ordered_files = self._download_data()
            self.dataproxy = self.load_data(
                ordered_files, atomrefs, uncharacterized
            )
            self.update_meta()
        else:
            self.dataproxy = self.load_dataproxy(self.datapath, self.format)

        return self.dataproxy

    def get_loader(self):
        if not self.is_prepared:
            self.prepare()
        return self.dataproxy.get_loader()

    def get_train_loader(self):
        return self.dataproxy.get_train_loader()

    def get_valid_loader(self):
        return self.dataproxy.get_val_loader()

    def get_test_loader(self):
        return self.dataproxy.get_test_loader()

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

        log.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        log.info("Done.")

        log.info("Parse xyz files...")
        ordered_files = sorted(
            raw_path.rglob("*.xyz"),
            key=lambda x: (int(re.sub("\D", "", str(x))), str(x)),
        )
        return ordered_files

    def load_data(self, files, atomrefs, uncharacterized) -> DataProxy:
        dataproxy = DataProxy()

        irange = np.arange(len(files), dtype=int)
        if uncharacterized is not None:
            irange = np.setdiff1d(
                irange, np.array(uncharacterized, dtype=int) - 1
            )

        for i in tqdm(irange):
            xyzfile = files[i]
            properties = {}

            tmp = io.StringIO()
            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for prop, p in zip(self.properties, l):
                    properties[prop] = molpy.units.convert(
                        (float(p), None), self.properties[prop]
                    )
                for line in lines:
                    tmp.write(line.replace("*^", "e"))

            # tmp.seek(0)
            # # ats: Atoms = list(read_xyz(tmp, 0))[0]
            # # properties[structure.Z] = ats.numbers
            # # properties[structure.R] = ats.positions
            # # properties[structure.cell] = ats.cell
            # # properties[structure.pbc] = ats.pbc
            # # property_list.append(properties)
            # frame =
            frame = mp.DataReader(xyzfile, "XYZ").read_frame()
            for k in self.keywords:
                properties[k.alias] = molpy.units.convert(
                    frame[k.keyword], k.unit, kw.get_unit(k.alias)
                )
            properties[kw.Z] = frame[kw.Z]
            properties[kw.R] = frame["positions"]
            properties[kw.cell] = frame.cell.tolist()
            properties[kw.pbc] = frame.cell.isPBC

            dataproxy[i] = properties

        return dataproxy
