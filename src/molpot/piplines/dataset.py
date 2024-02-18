from io import BytesIO, TextIOWrapper
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from pathlib import Path
import logging
import numpy as np
import torch
import tempfile
from itertools import islice
import molpy as mp
from molpot import alias


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
        name: str,
        save_dir: None | Path | str,
        total: int,
        batch_size: int,
        device: str = "cpu",
    ):
        self.name = name
        self.total = total
        self.logger = logging.getLogger(self.name)
        self.batch_size = batch_size
        self.in_memory = True if save_dir is None else False
        self.save_dir = save_dir
        self.device = device

    def save_to(self, url):
        basename = Path(url).name
        if self.save_dir is None:
            save_dir = Path(tempfile.gettempdir()) / basename
        else:
            save_dir = Path(self.save_dir) / basename
        return str(save_dir)

    def _prepare(self, dp: IterDataPipe) -> IterDataPipe:
        dp = dp.header(self.total).set_length(self.total).pin_memory(device=self.device)
        return dp.batch(self.batch_size)


def read_stream_as_text(_tuple: tuple[str, bytes]):
    path, stream = _tuple
    return path, TextIOWrapper(BytesIO(stream.read()))


def read_stream_as_bytes(_tuple: tuple[str, bytes]):
    path, stream = _tuple
    return path, BytesIO(stream.read())


class QM9(DataSet):
    def __init__(
        self,
        save_dir: None | Path | str = None,
        total: int = 0,
        batch_size: int = 1,
        device: str = "cpu",
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("QM9", save_dir, total, batch_size, device)
        self.remove_uncharacterized = remove_uncharacterized
        self.atom_ref = atom_ref
        alias("QM9")
        alias.QM9.set("A", "_A", float, "GHz", "rotational_constant_A")
        alias.QM9.set("B", "_B", float, "GHz", "rotational_constant_B")
        alias.QM9.set("C", "_C", float, "GHz", "rotational_constant_C")
        alias.QM9.set("mu", "_mu", float, "Debye", "dipole_moment")
        alias.QM9.set("alpha", "_alpha", float, "a0 a0 a0", "isotropic_polarizability")
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

    def prepare(self, device="cpu") -> IterDataPipe:

        url = "https://ndownloader.figshare.com/files/3195389"  # tar.bz2

        cache_dp = (
            IterableWrapper([url])
            .on_disk_cache(filepath_fn=self.save_to)
            .read_from_http()
            .load_from_bz2(length=self.total)
            .load_from_tar(length=self.total)
        )

        cache_dp.end_caching(same_filepath_fn=True)
        dp = cache_dp.map(read_stream_as_text).read_qm9()

        return self._prepare(dp, device)

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


class RMD17(DataSet):
    def __init__(
        self,
        save_dir: None | Path | str = None,
        total: int = 0,
        batch_size: int = 64,
        device: str = "cpu",
        molecule: str = "aspirin",
    ):
        super().__init__("rmd17", save_dir, total, batch_size, device)
        self.molecule = molecule
        alias("rmd17")
        alias.rmd17.set("molecule", "_rmd17_molecule", str, None, "molecule name")
        alias.rmd17.set("energy", "_rmd17_U", float, "kcal/mol", "_energy")
        alias.rmd17.set("forces", "_rmd17_F", float, "kcal/mol/angstrom", "_forces")
        alias.rmd17.set("R", "_rmd17_R", np.ndarray, "angstrom", "atomic coordinates")
        alias.rmd17.set("Z", "_rmd17_Z", int, None, "atomic numbers in molecule")

    def get_molecule(self, _tuple: str):
        filename = _tuple[0]
        if filename.endswith(f"{self.molecule}.npz"):
            print(filename)
            return True
        return False

    def prepare(self) -> IterDataPipe:
        url = "https://figshare.com/ndownloader/files/23950376"

        dp = IterableWrapper([url])
        cache_dp = (
            dp.on_disk_cache(filepath_fn=self.save_to)
            .read_from_http()
            .load_from_bz2(length=self.total)
            .load_from_tar(length=self.total)
            .filter(filter_fn=self.get_molecule)
        )

        dp = cache_dp.end_caching(same_filepath_fn=True).read_rmd17()

        return self._prepare(dp)


class Trajectory(DataSet):

    def __init__(
        self, trajectory: mp.io.TrajLoader, total: int = 0, batch_size: int = 1
    ):
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
