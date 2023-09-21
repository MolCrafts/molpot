# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-21
# version: 0.0.1

import io
import logging
from pathlib import Path
import tempfile
from torch.utils.data import Dataset

from typing import Any, Optional
from urllib.request import urlretrieve
import numpy as np

import tarfile
import re
import os
import tqdm

from molpot import DataSet, DataProxy, DataLoader

log = logging.getLogger(__name__)

class Keywords:

    def __init__(self, name="_mpot_global"):

        self._maps:dict[str, str] = {}
        self._units:dict[str, str] = {}

    def __getitem__(self, key):
        return self._maps[key]
    
    def __setitem__(self, key, value):
        self._maps[key] = value

    def __getattribute__(self, __name: str) -> str:
        return self._maps[__name]
    
    def __setattr__(self, __name: str, value: str) -> None:
        self._maps[__name] = value

    def get_unit(self, key):
        return self._units[key]
    
    def set_unit(self, key, value):
        self._units[key] = value

    def set(self, alias, keyword, unit):
        self._maps[alias] = keyword
        self._units[alias] = unit

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

    def __init__(self, name, data_dir: Optional[Path | str], batch_size: int, n_train: Optional[int], n_valid: Optional[int], n_test: Optional[int], loader: DataLoader):

        if self.data_dir is None:
            self.data_dir = tempfile.mkdtemp()
        else:
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)

        self.keywords = Keywords(name)

    def set_keywords(self, keywords: dict[str, str]):
        self.keywords = keywords

    def prepare(self):
        raise NotImplementedError

    def fetch(self, url, filename, dir=Optional[Path | str])->Path:
        if dir is None:
            dir = self.data_dir
        log.info(f'downloading from {url}...')
        fpath = Path(dir)/Path(filename)
        urlretrieve(url, fpath)
        return fpath
    
    def create_dataproxy(self, format) -> DataProxy:
        if format == "ASE":
            return ASEDataProxy()
    
    def load_dataproxy(self, format) -> DataProxy:
        if format == "ASE":
            return ASEDataProxy()
    
class QM9DataSet(DataSet):

    def __init__(self, data_dir: Optional[Path | str], batch_size: int, n_train: Optional[int], n_valid: Optional[int], n_test: Optional[int], loader: DataLoader, remove_uncharacterized: bool = True):
        super().__init__("QM9", data_dir, batch_size, n_train, n_valid, n_test, loader)
        self.remove_uncharacterized = remove_uncharacterized
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

    def prepare(self) -> DataProxy:
        
        if not self.data_dir / Path('data.meta').exists():
            atomrefs = self._download_atomrefs()
            # dataproxy = self.create_dataproxy()
            if self.remove_uncharacterized:
                uncharacterized = self._download_uncharacterized()
            else:
                uncharacterized = None
            self._download_data()
            self.dataproxy = self.dataloader(self.data_dir)

        else:
            self.dataproxy = self.load_dataproxy(self.datapath, self.format)
            
        return dataproxy
    
    def get_dataset(self):
        return dataproxy.get_dataset()
        
    def _download_atomrefs(self):
        url = "https://ndownloader.figshare.com/files/3195395"
        filename = 'atomref.txt'
        atomrefs_path = self.fetch(url, filename)
        props = [self.zpve, self.U0, self.U, self.H, self.G, self.Cv]
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
    
    def _download_data(
        self, dataproxy: DataProxy, uncharacterized: Optional[list[int]]
    ):
        url = "https://ndownloader.figshare.com/files/3195389"
        tar_path = self.fetch(url, "gdb9.tar.gz")
        raw_path = self.fetch(url, "gdb9_xyz")

        log.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        log.info("Done.")

        log.info("Parse xyz files...")
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
        )
        return ordered_files

        # property_list = []

        # irange = np.arange(len(ordered_files), dtype=int)
        # if uncharacterized is not None:
        #     irange = np.setdiff1d(irange, np.array(uncharacterized, dtype=int) - 1)

        # for i in tqdm(irange):
        #     xyzfile = os.path.join(raw_path, ordered_files[i])
        #     properties = {}

        #     tmp = io.StringIO()
        #     with open(xyzfile, "r") as f:
        #         lines = f.readlines()
        #         l = lines[1].split()[2:]
        #         for pn, p in zip(data.available_properties, l):
        #             properties[pn] = np.array([float(p)])
        #         for line in lines:
        #             tmp.write(line.replace("*^", "e"))

        #     tmp.seek(0)
        #     ats: Atoms = list(read_xyz(tmp, 0))[0]
        #     properties[structure.Z] = ats.numbers
        #     properties[structure.R] = ats.positions
        #     properties[structure.cell] = ats.cell
        #     properties[structure.pbc] = ats.pbc
        #     property_list.append(properties)


        # log.info("Write atoms to db...")
        # dataset.add_systems(property_list=property_list)
        # log.info("Done.")