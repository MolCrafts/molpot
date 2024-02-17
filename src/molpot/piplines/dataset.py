from functools import partial
from io import BytesIO
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper, HttpReader
from torchdata.dataloader2 import DataLoader2
from typing import Optional, Any
from pathlib import Path
import logging
import requests
import numpy as np
import torch
import tarfile
from itertools import islice
import molpy as mp
from molpot import alias
import zipfile
import random

class DataSet:

    """
    Base class for all datasets. It includes 5 processes:
        * Download / tokenize / process.
        * Clean and (maybe) save to disk.
        * Load inside Dataset.
        * Apply transforms (rotate, tokenize, etc…).
        * Wrap inside a DataLoader.
    """

    def __init__(self, name:str, save_dir: None | Path | str, total: int, batch_size: int):
        self.name = name
        self.total = total
        self.logger = logging.getLogger(self.name)
        self.batch_size = batch_size
        self.in_memory = True if save_dir is None else False
        if save_dir is not None:
            self.init_save_dir(Path(save_dir))

    def init_save_dir(self, save_dir: Path):
        if save_dir.exists():
            pass
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
        # self.update_meta()

    def _prepare(self, dp: IterDataPipe) -> IterDataPipe:
        # if self.total:
        #     dp = dp.header(self.total).set_length(self.total)
        # return dp.batch(self.batch_size)
        return dp.batch(self.batch_size)

    # def download(self, url, fpath: Path) -> Path:
    #     response = requests.get(url, stream=True)
    #     assert response.status_code == 200, f"Failed to download {url}"
    #     with open(fpath, "wb") as f:
    #         for chunk in response.iter_content(chunk_size=1024):
    #             if chunk:
    #                 f.write(chunk)
    #                 f.flush()

    #     return fpath
        
    # def load(self, url) -> bytes:
    #     response = requests.get(url)
    #     assert response.status_code == 200, f"Failed to download {url}"
    #     return response.content
    

class QM9(DataSet):
    def __init__(
        self,
        save_dir: None | Path | str = None,
        total: int = 0,
        batch_size: int = 1,
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("QM9", save_dir, total, batch_size)
        self.remove_uncharacterized = remove_uncharacterized
        self.atom_ref = atom_ref
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

        url = 'https://ndownloader.figshare.com/files/3195389'  # tar.bz2

        if self.in_memory:
            http_reader_dp = HttpReader(IterableWrapper([url]))
            dp = http_reader_dp.load_from_bz2(length=self.total).load_from_tar(length=self.total)  # (filename, StreamWrapper)

        dp.read_qm9()
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
        alias.rMD17.set("energy", "_rmd17_U", float, "kcal/mol", "_energy")
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