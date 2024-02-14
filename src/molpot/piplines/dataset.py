from functools import partial
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchdata.dataloader2 import DataLoader2
from typing import Optional, Any
from pathlib import Path
import tempfile
import time
import json
import logging
from urllib.request import urlopen
import numpy as np
import torch
import tarfile
from itertools import islice
import molpy as mp
from molpot import alias
import zipfile

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
        self, name, data_dir: None | Path | str, in_memory: bool = True, total: Optional[None] = None, batch_size: int = 1
    ):
        super().__init__()
        self.name = name
        self.in_memory = in_memory
        self.total = total
        self.logger = logging.getLogger(self.name)
        self.batch_size = batch_size
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
            dp = dp.header(self.total).set_length(self.total)
        return dp.batch(self.batch_size)

    def fetch(self, url, filename, dir: Path | str | None) -> Path | bytes:
        # fpath = Path(dir) / Path(filename)
        # if dir is None:
        #     dir = self.data_dir
        # if not fpath.exists():
        #     log.info(f"downloading from {url}... to {fpath}")
        #     response = urlopen(url, fpath)
        # else:
        #     log.info(f"{fpath} already exists.")
        # return fpath
        with urlopen(url) as response:
            byte_data = response.read()
            if dir is None:
                return byte_data
            else:
                dir = Path(dir)
                dir.mkdir(parents=True, exist_ok=True)
                with open(dir / filename, "wb") as f:
                    f.write(byte_data)
                return dir / filename

    def extract_tar(self, tar_path, dest_path, member):

        if dest_path.exists():
            log.info(f"{dest_path} already exists.")
            path = dest_path
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
        batch_size: int = 1,
    ):
        super().__init__("QM9", data_dir, in_memory, total, batch_size)
        self.remove_uncharacterized = remove_uncharacterized
        alias("QM9")
        alias.QM9.set("A", "_A", float, "GHz", "rotational_constant_A")
        alias.QM9.set("B", "_B", float, "GHz", "rotational_constant_B")
        alias.QM9.set("C", "_C", float, "GHz", "rotational_constant_C")
        alias.QM9.set("mu", "_mu", float, "Debye", "dipole_moment")
        alias.QM9.set(
            "alpha", "_alpha", float, "a0 a0 a0", "isotropic_polarizability"
        )
        alias.QM9.set("homo", "_homo", float, "hartree", "homo")
        alias.QM9.set("lumo", "_lumo", float, "hartree", "lump")
        alias.QM9.set("gap", "_gap", float, "hartree", "gap")
        alias.QM9.set("r2", "_r2", float, "a0 a0", "electronic_spatial_extent")
        alias.QM9.set("zpve", "_zpve", float, "hartree", "zpve")
        alias.QM9.set("U0", "_U0", float, "hartree", "_energy_U0")
        alias.QM9.set("U", "_U", float, "hartree", "_energy_U")
        alias.QM9.set("H", "_H", float, "hartree", "_enthalpy_H")
        alias.QM9.set("G", "_G", float, "hartree", "_free_energy")
        alias.QM9.set("Cv", "_Cv", float, "cal/mol/K", "_heat_capacity")

    def prepare(self) -> IterDataPipe:

        # atomrefs = self._download_atomrefs()
        # if self.remove_uncharacterized:
        #     uncharacterized = self._download_uncharacterized()
        # else:
        #     uncharacterized = None

        if self.in_memory:
            # import requests
            # import io

            # qm9_url = "https://ndownloader.figshare.com/files/3195389"
            # qm9_bytes = requests.get(qm9_url, allow_redirects=True).content
            # qm9_fobj = io.BytesIO(qm9_bytes)
            # qm9_fobj.seek(0)
            # qm9_tar = tarfile.open(fileobj=qm9_fobj, mode="r:bz2")
            # names = qm9_tar.getnames()

            # exclude_url = "https://figshare.com/ndownloader/files/3195404"
            # exclude_bytes = requests.get(exclude_url, allow_redirects=True).content
            # exclude_fobj = io.TextIOWrapper(io.BytesIO(exclude_bytes))
            # exclude = [int(line.split()[0]) for line in exclude_fobj.readlines()[9:-1]]
            # names = [name for name in names if int(name[-10:-4]) not in exclude]
            # dp = (
            #     IterableWrapper(names)
            #     .map(qm9_tar.extractfile)
            # )
            # return self._prepare(dp)
            raise NotImplementedError('Can not read from memory, torchdata not allow to pickle _io.BufferedReader')

        else:

            filepaths = self._download_data()

            dp = (
                IterableWrapper(list(filepaths))
                .open_files()
            )
        dp = dp.read_qm9()
        return self._prepare(dp)

    def _download_atomrefs(self):
        url = "https://ndownloader.figshare.com/files/3195395"
        filename = "atomref.txt"
        atomrefs_path = self.fetch(url, filename, self.data_dir)
        props = [
            alias.QM9.zpve,
            alias.QM9.U0,
            alias.QM9.U,
            alias.QM9.H,
            alias.QM9.G,
            alias.QM9.Cv,
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
        dest_path = self.data_dir / Path("xyz")

        # _sort_qm9 = lambda x: (int(re.sub(r"\D", "", str(x))), str(x))
        self.extract_tar(tar_path, dest_path, "all")

        log.info("Parse xyz files...")
        xyz_files = dest_path.rglob("*.xyz")
        xyz_files = list(map(str, xyz_files))
        return xyz_files

class rMD17(DataSet):
    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        in_memory: bool = False,
        total: int = 0,
        batch_size: int = 64,
        molecule: str = "aspirin",
    ):
        super().__init__("rMD17", data_dir, False, total)
        self.molecule = molecule
        self.batch_size = batch_size
        alias("rMD17")
        alias.rMD17.set("U", "_rmd17_U", float, "kcal/mol", "_energy")
        alias.rMD17.set(
            "forces", "_rmd17_F", float, "kcal/mol/angstrom", "_forces"
        )
        alias.rMD17.set(
            "R", "_rmd17_R", np.ndarray, "angstrom", "atomic coordinates"
        )
        alias.rMD17.set("Z", "_rmd17_Z", int, None, "atomic numbers in molecule")
        self.in_memory = in_memory

    def prepare(self) -> IterDataPipe:
        byte_data = self._download_data()
        dp = IterableWrapper([fpath]).read_rmd17()
        return self._prepare(dp)

    def _download_data(
        self,
    )-> bytes:
        
#   rmd17_url = "https://figshare.com/ndownloader/articles/12672038/versions/3"
#   zip_bytes = requests.get(rmd17_url, allow_redirects=True).content
#   zip_bytes_io = io.BytesIO(zip_bytes)
#   zip_bytes_io.seek(0)
#   zip_fobj = zipfile.ZipFile(zip_bytes_io)
#   tar_path = zip_fobj.extract(zip_fobj.infolist()[0])
#   tar_fobj = tarfile.open(tar_path)
#   npz_obj = tar_fobj.extractfile('rmd17/npz_data/rmd17_${rmd17_tag}.npz')
#   tag_npz = np.load(npz_obj)
#   tag_size = tag_npz['coords'].shape[0]
#   tag_ds = load_numpy({
#       'elems': np.repeat(tag_npz['nuclear_charges'][None,:], tag_size, axis=0),
#       'coord': tag_npz['coords'],
#       'e_data': tag_npz['energies'],
#       'f_data': tag_npz['forces']
#   })
#   write_tfrecord(f'rmd17-${rmd17_tag}.yml', tag_ds)
#   tar_fobj.close()
#   zip_fobj.close()
#   os.remove(tar_path)

        logging.info("Downloading {} data".format(self.molecule))
        dest_path = self.data_dir / 'npz_data'
        tar_path = self.data_dir / Path("rmd17.tar.gz")
        url = "https://figshare.com/ndownloader/articles/12672038/versions/3"
        if self.in_memory:
            byte_data = self.fetch(url, None, None)
            zip_fobj = zipfile.ZipFile(byte_data)
            tar_path = zip_fobj.extract(zip_fobj.infolist()[0])
            tar_fobj = tarfile.open(tar_path)
            npz_obj = tar_fobj.extractfile('rmd17/npz_data/rmd17_${rmd17_tag}.npz')
        else:
            tar_path = self.fetch(url, "rmd17.tar.gz", self.data_dir)

            npz_obj = self.extract_tar(tar_path, dest_path, f"rmd17/npz_data/rmd17_{self.molecule}.npz")
        
        logging.info("Parsing molecule {:s}".format(self.molecule))

        return npz_obj

class Trajectory(DataSet):

    def __init__(self, trajectory:mp.io.TrajLoader, total: int = 0, batch_size: int = 1):
        super().__init__("Trajectory", None, True, total, batch_size)
        self.trajectory = trajectory

    def prepare(self) -> IterDataPipe:
        # Since ctypes objects containing pointers cannot be pickled
        # pre-load instead of lazy loading
        frames = [frame.as_dict() for frame in islice(self.trajectory, self.total)]
        frames = [{k: torch.tensor(v) for k, v in frame.items()} for frame in frames]
        dp = IterableWrapper(frames)
        self.trajectory.close()
        del self.trajectory
        return self._prepare(dp)