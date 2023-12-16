from functools import partial
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchdata.dataloader2 import DataLoader2
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
import molpy as mp
from itertools import islice

__all__ = ["DataSet", "DataLoader2", "QM9", "rMD17"]


log = logging.getLogger(__name__)


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
        self, name, data_dir: None | Path | str, in_memory: bool = True, total: int = 0
    ):
        super().__init__()
        self.name = name
        self.in_memory = in_memory
        self.total = total
        if not in_memory:
            if data_dir:
                self.data_dir = Path(data_dir)
            else:
                self.data_dir = Path(tempfile.mkdtemp())
            if not self.data_dir.exists():
                self.data_dir.mkdir(parents=True, exist_ok=True)

    def update_meta(self, data: Optional[dict[str, Any]] = None):
        self._data = {
            "name": self.name,
            "update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        self._data.update(data or {})
        if not self.in_memory:
            with open(self.data_dir / Path("meta.json"), "w") as f:
                json.dump(self._data, f)

    def _prepare(self, dp) -> IterDataPipe:
        if self.total:
            dp.set_length(self.total)
        return dp

    def fetch(self, url, filename, dir: Path | str) -> Path:
        fpath = Path(dir) / Path(filename)
        if dir is None:
            dir = self.data_dir
        if not fpath.exists():
            log.info(f"downloading from {url}... to {fpath}")
            urlretrieve(url, fpath)
        else:
            log.info(f"{fpath} already exists.")
        return fpath

    def extract_tar(self, tar_path, dest_path, member):

        path = dest_path / member

        if path.exists():
            log.info(f"{dest_path} already exists.")
        else:
            log.info("Extracting files...")
            tar = tarfile.open(tar_path)
            if member == "all":
                path = tar.extractall(dest_path)
            else:
                path = tar.extract(path=dest_path, member=str(member))
            tar.close()
            log.info("Done.")
        return path


class QM9(DataSet):
    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        in_memory: bool = False,
        remove_uncharacterized: bool = True,
        total: int = 0,
    ):
        super().__init__("QM9", data_dir, in_memory, total)
        self.remove_uncharacterized = remove_uncharacterized
        mpot.alias("QM9")
        mpot.alias.QM9.set("A", "_A", float, "GHz", "rotational_constant_A")
        mpot.alias.QM9.set("B", "_B", float, "GHz", "rotational_constant_B")
        mpot.alias.QM9.set("C", "_C", float, "GHz", "rotational_constant_C")
        mpot.alias.QM9.set("mu", "_mu", float, "Debye", "dipole_moment")
        mpot.alias.QM9.set(
            "alpha", "_alpha", float, "a0 a0 a0", "isotropic_polarizability"
        )
        mpot.alias.QM9.set("homo", "_homo", float, "hartree", "homo")
        mpot.alias.QM9.set("lumo", "_lumo", float, "hartree", "lump")
        mpot.alias.QM9.set("gap", "_gap", float, "hartree", "gap")
        mpot.alias.QM9.set("r2", "_r2", float, "a0 a0", "electronic_spatial_extent")
        mpot.alias.QM9.set("zpve", "_zpve", float, "hartree", "zpve")
        mpot.alias.QM9.set("U0", "_U0", float, "hartree", "_energy_U0")
        mpot.alias.QM9.set("energy", "_U", float, "hartree", "_energy_U")
        mpot.alias.QM9.set("H", "_H", float, "hartree", "_enthalpy_H")
        mpot.alias.QM9.set("G", "_G", float, "hartree", "_free_energy")
        mpot.alias.QM9.set("Cv", "_Cv", float, "cal/mol/K", "_heat_capacity")


    def prepare(self) -> IterDataPipe:
        if self.in_memory:
            import requests
            import io

            qm9_url = "https://ndownloader.figshare.com/files/3195389"
            qm9_bytes = requests.get(qm9_url, allow_redirects=True).content
            qm9_fobj = io.BytesIO(qm9_bytes)
            qm9_fobj.seek(0)
            qm9_tar = tarfile.open(fileobj=qm9_fobj, mode="r:bz2")
            names = qm9_tar.getnames()

            exclude_url = "https://figshare.com/ndownloader/files/3195404"
            exclude_bytes = requests.get(exclude_url, allow_redirects=True).content
            exclude_fobj = io.TextIOWrapper(io.BytesIO(exclude_bytes))
            exclude = [int(line.split()[0]) for line in exclude_fobj.readlines()[9:-1]]
            names = [name for name in names if int(name[-10:-4]) not in exclude]
            dp = (
                IterableWrapper(names)
                .map(lambda x: io.TestIOWrapper(qm9_tar.extractfile(x)).readlines())
                .read_qm9()
            )

        else:
            # atomrefs = self._download_atomrefs()
            if self.remove_uncharacterized:
                uncharacterized = self._download_uncharacterized()
            else:
                uncharacterized = None
            ordered_files = self._download_data()

            irange = np.arange(len(ordered_files), dtype=int)
            if uncharacterized is not None:
                irange = np.setdiff1d(irange, np.array(uncharacterized, dtype=int) - 1)

            dp = (
                IterableWrapper(map(str, np.array(ordered_files)[irange]))
                .open_files()
                .read_qm9()
            )
        return self._prepare(dp)

    def _download_atomrefs(self):
        url = "https://ndownloader.figshare.com/files/3195395"
        filename = "atomref.txt"
        atomrefs_path = self.fetch(url, filename, self.data_dir)
        props = [
            mpot.alias.QM9.zpve,
            mpot.alias.QM9.U0,
            mpot.alias.QM9.U,
            mpot.alias.QM9.H,
            mpot.alias.QM9.G,
            mpot.alias.QM9.Cv,
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
        path = self.fetch(at_url, "uncharacterized.txt", self.data_dir)
        uncharacterized = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_data(self):
        url = "https://ndownloader.figshare.com/files/3195389"
        tar_path = self.fetch(url, "gdb9.tar.gz", self.data_dir)
        raw_path = self.data_dir / Path("gdb9_xyz")

        _sort_qm9 = lambda x: (int(re.sub(r"\D", "", str(x))), str(x))
        self.extract_tar(tar_path, raw_path, "all")

        log.info("Parse xyz files...")
        xyz_files = raw_path.rglob("*.xyz")
        if self.test_size:
            xyz_files = islice(xyz_files, self.test_size)
        ordered_files = sorted(
            xyz_files,
            key=_sort_qm9,
        )  # sort by index in filename
        return ordered_files


class rMD17(DataSet):
    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        total: int = 0,
        molecule: str = "aspirin",
    ):
        super().__init__("rMD17", data_dir, False, total)
        self.molecule = molecule
        mpot.alias("rMD17")
        mpot.alias.rMD17.set("energy", "_rmd17_U", float, "kcal/mol", "_energy_U")
        mpot.alias.rMD17.set(
            "forces", "_rmd17_F", float, "kcal/mol/angstrom", "_forces"
        )
        mpot.alias.rMD17.set(
            "R", "_rmd17_R", np.ndarray, "angstrom", "atomic coordinates"
        )
        mpot.alias.rMD17.set("Z", "_rmd17_Z", int, None, "atomic numbers in molecule")

    def prepare(self) -> IterDataPipe:
        fpath = self._download_data()
        dp = IterableWrapper([fpath]).read_rmd17()
        return self._prepare(dp)

    def _download_data(
        self,
    ):
        logging.info("Downloading {} data".format(self.molecule))
        dest_path = self.data_dir
        tar_path = self.data_dir / Path("rmd17.tar.gz")
        url = "https://figshare.com/ndownloader/files/23950376"
        self.fetch(url, "rmd17.tar.gz", self.data_dir)
        logging.info("Done.")
        path = self.extract_tar(tar_path, dest_path, f"rmd17/npz_data/rmd17_{self.molecule}.npz")
        logging.info("Parsing molecule {:s}".format(self.molecule))

        return path
