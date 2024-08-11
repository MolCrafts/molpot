import logging
import tempfile
from io import BytesIO, TextIOWrapper
from pathlib import Path

import torch

from torch.utils.data import IterableDataset as BaseDataset
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from .process import apply_dress, atomic_dress
from molpot.alias import Alias, NameSpace


class DataSet(BaseDataset):
    """
    Base class for all datasets. It includes 5 processes:
        * Download / tokenize / process.
        * Clean and (maybe) save to disk.
        * Load inside Dataset.
        * Apply transforms (rotate, tokenize, etc…).
        * Wrap inside a DataLoader.
    """

    labels: NameSpace

    def __init__(
        self,
        name: str,
        save_dir: None | Path | str,
        total: int,
        device: str = "cpu",
    ):
        self.name = name
        self.total = total
        self.device = device

        if save_dir is None:
            self.save_dir = Path(tempfile.gettempdir()) / self.name
        else:
            self.save_dir = Path(save_dir) / self.name

        self.labels = NameSpace(name)

    def get_pipeline(self) -> IterDataPipe:
        pass

    def __len__(self):
        return self.total


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
        device: str = "cpu",
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("qm9", save_dir, total, device)
        self.remove_uncharacterized = remove_uncharacterized
        self.atom_ref = atom_ref
        self.labels.set(
            "A", ("targets", "A"), float, "GHz", "rotational_constant_A", shape=()
        )
        self.labels.set(
            "B", ("targets", "B"), float, "GHz", "rotational_constant_B", shape=()
        )
        self.labels.set(
            "C", ("targets", "C"), float, "GHz", "rotational_constant_C", shape=()
        )
        self.labels.set(
            "mu", ("targets", "mu"), float, "Debye", "dipole_moment", shape=()
        )
        self.labels.set(
            "alpha",
            ("targets", "alpha"),
            float,
            "a0",
            "isotropic_polarizability",
            shape=(),
        )
        self.labels.set("homo", ("targets", "homo"), float, "hartree", "homo", shape=())
        self.labels.set("lumo", ("targets", "lumo"), float, "hartree", "lump", shape=())
        self.labels.set("gap", ("targets", "gap"), float, "hartree", "gap", shape=())
        self.labels.set(
            "r2", ("targets", "r2"), float, "a0", "electronic_spatial_extent", shape=()
        )
        self.labels.set("zpve", ("targets", "zpve"), float, "hartree", "zpve", shape=())
        self.labels.set("U0", ("targets", "U0"), float, "hartree", "energy_U0", shape=())
        self.labels.set("U", ("targets", "U"), float, "hartree", "energy_U", shape=())
        self.labels.set("H", ("targets", "H"), float, "hartree", "enthalpy_H", shape=())
        self.labels.set("G", ("targets", "G"), float, "hartree", "free_energy", shape=())
        self.labels.set(
            "Cv", ("targets", "Cv"), float, "cal/mol/K", "heat_capacity", shape=()
        )

    def get_pipeline(self) -> IterDataPipe:

        url = "https://ndownloader.figshare.com/files/3195389"  # tar.bz2

        dp = (
            IterableWrapper([url])
            .read_from_http()
            .load_from_bz2(length=self.total)
            .load_from_tar(length=self.total)
            .read_from_stream()
            .read_qm9()
        )
        if self.total > 0:
            dp = dp.header(self.total).set_length(self.total)
        return dp

    def __iter__(self):
        dp = self.get_pipeline()
        for d in dp:
            yield d

    def __len__(self):
        return self.total

    @property
    def Z(self):
        return torch.tensor([1, 6, 7, 8, 9])

    @property
    def atomrefs(self):
        lines = self._download_atomrefs()
        props = [
            self.labels["zpve"],
            self.labels["U0"],
            self.labels["U"],
            self.labels["H"],
            self.labels["G"],
            self.labels["Cv"],
        ]
        atref = {z.item(): {} for z in self.Z}
        for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
            for i, p in enumerate(props):
                atref[z][p] = float(l.split()[i + 1])
        return atref

    def _download_atomrefs(self):
        url = IterableWrapper(["https://ndownloader.figshare.com/files/3195395"])
        dp = url.read_from_http().readlines()
        lines = []
        for _, line in dp:
            lines.append(line.decode("utf-8"))
        return lines

    def _download_uncharacterized(self):
        at_url = "https://ndownloader.figshare.com/files/3195404"
        path = self.fetch(at_url, "uncharacterized.txt", self.data_dir)
        uncharacterized = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized
