import io
import logging
import tarfile
import time
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset

import molpot as mpot
from molpot import NameSpace, alias


class IterableDataset(torch.utils.data.IterableDataset):
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
        save_dir: str | Path,
        device: str = "cpu",
    ):
        self.name = name
        self.save_dir = Path(save_dir)
        self.device = device
        self.labels = NameSpace(name)
        self.logger = logging.getLogger(__name__)

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def reset(self):
        self.state = {}


class QM9(IterableDataset):
    def __init__(
        self,
        save_dir: str | Path,
        device: str = "cpu",
        total: int = 133885,
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("qm9", save_dir, device)
        self.remove_uncharacterized = remove_uncharacterized
        self.atom_ref = atom_ref

        self.labels.set("A", float, "GHz", "rotational_constant_A")
        self.labels.set("B", float, "GHz", "rotational_constant_B")
        self.labels.set("C", float, "GHz", "rotational_constant_C")
        self.labels.set("mu", float, "Debye", "dipole_moment")
        self.labels.set("alpha", float, "a0", "isotropic_polarizability")
        self.labels.set("alpha", float, "a0", "isotropic_polarizability")
        self.labels.set("homo", float, "hartree", "homo")
        self.labels.set("lumo", float, "hartree", "lump")
        self.labels.set("gap", float, "hartree", "gap")
        self.labels.set("r2", float, "a0", "electronic_spatial_extent")
        self.labels.set("zpve", float, "hartree", "zpve")
        self.labels.set("U0", float, "hartree", "energy_U0")
        self.labels.set("U", float, "hartree", "energy_U")
        self.labels.set("H", float, "hartree", "enthalpy_H")
        self.labels.set("G", float, "hartree", "free_energy")
        self.labels.set("Cv", float, "cal/mol/K", "heat_capacity")

        self.reset()
        self.total = total

        self.frames = self._download_data(total)

    # @property
    # def data(self):
    #     return self._download_data()

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        frames = self.frames
        for frame in frames:
            yield frame
        self.reset()

    # def _download_uncharacterized(self):
    #     self.logger.info("Downloading list of uncharacterized molecules...")
    #     at_url = "https://ndownloader.figshare.com/files/3195404"
    #     requests.get(at_url).content
    #     self.logger.info("Done.")

    #     uncharacterized = []
    #     with open(io.TextIOWrapper(io.BytesIO())) as f:
    #         lines = f.readlines()
    #         for line in lines[9:-1]:
    #             uncharacterized.append(int(line.split()[0]))
    #     return uncharacterized

    # def _download_atomrefs(self, tmpdir):
    #     self.logger.info("Downloading GDB-9 atom references...")
    #     at_url = "https://ndownloader.figshare.com/files/3195395"
    #     atomrefs = requests.get(at_url).content

    #     qm9 = self.labels

    #     props = [qm9.zpve, qm9.U0, qm9.U, qm9.H, qm9.G, qm9.Cv]
    #     atref = {p: torch.zeros((100,)) for p in props}
    #     with open(atomrefs) as f:
    #         lines = f.readlines()
    #         for z, l in zip([1, 6, 7, 8, 9], lines[5:10]):
    #             for i, p in enumerate(props):
    #                 atref[p][z] = float(l.split()[i + 1])
    #     atref = {k: v.tolist() for k, v in atref.items()}
    #     return atref

    def _download_data(self, total):
        self.logger.info("Downloading GDB-9 data...")
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
        QM9 = self.labels
        props = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]

        frames = []
        time_pt = time.perf_counter()
        for i, name in enumerate(names):
            f = io.TextIOWrapper(qm9_tar.extractfile(name))
            lines = f.readlines()
            Z = [mpot.Element[l.split()[0]].number for l in lines[2:-3]]
            R = [[float(i.replace("*^", "E")) for i in l.split()[1:4]] for l in lines[2:-3]]
            frame = mpot.Frame()
            frame[alias.Z] = torch.tensor(Z)
            frame[alias.R] = torch.tensor(R, dtype=torch.float)
            frame[alias.n_atoms] = torch.tensor(len(Z))
            prop_line = lines[1].split()
            for k, v in zip(props, prop_line[1:]):
                frame[k] = torch.tensor(float(v))
            frames.append(frame)
            end_time = time.perf_counter()
            self.logger.debug(f"Frame {i} loaded in {end_time - time_pt:.2f}s")
            time_pt = end_time
            if i == total:
                break
        return frames
