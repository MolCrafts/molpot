import io
import logging
import random
import tarfile
import time
from functools import reduce
from pathlib import Path

import requests
import torch
from torch.nn import Module
from torch.utils.data import Dataset

import molpot as mpot
from molpot import NameSpace, alias

logger = logging.getLogger("molpot")

class Dataset:
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
        preprocess: list[Module] = []
    ):
        super().__init__()
        self.name = name
        self.save_dir = Path(save_dir)
        self.device = device
        self.labels = NameSpace(name)
        self._preprocess = preprocess

    def pre_process(self, inputs):
        for proc in self._preprocess:
            inputs = proc(inputs)
        return inputs

    def __len__(self):
        pass

    def __iter__(self):
        pass

    def reset(self):
        self.state = {}


class QM9(Dataset, torch.utils.data.Dataset):
    def __init__(
        self,
        save_dir: str | Path,
        device: str = "cpu",
        total: int = 133885,
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
        preprocess: list[Module] = []
    ):
        super().__init__("qm9", save_dir, device, preprocess=preprocess)

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

        self.total = total

        self.frames = self._download_data(total)

    def __len__(self):
        return self.total

    def __iter__(self):
        return iter(map(self.pre_process, self.frames.values()))

    def __getitem__(self, idx):
        return self.pre_process(self.frames.values()[idx])

    # def _download_uncharacterized(self):
    #     logger.info("Downloading list of uncharacterized molecules...")
    #     at_url = "https://ndownloader.figshare.com/files/3195404"
    #     requests.get(at_url).content
    #     logger.info("Done.")

    #     uncharacterized = []
    #     with open(io.TextIOWrapper(io.BytesIO())) as f:
    #         lines = f.readlines()
    #         for line in lines[9:-1]:
    #             uncharacterized.append(int(line.split()[0]))
    #     return uncharacterized

    # def _download_atomrefs(self, tmpdir):
    #     logger.info("Downloading GDB-9 atom references...")
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
        logger.info("start to download data...")
        qm9_url = "https://ndownloader.figshare.com/files/3195389"
        qm9_bytes = requests.get(qm9_url, allow_redirects=True).content
        qm9_fobj = io.BytesIO(qm9_bytes)
        qm9_fobj.seek(0)
        qm9_tar = tarfile.open(fileobj=qm9_fobj, mode="r:bz2")
        names = qm9_tar.getnames()
        logger.info(f"find {len(names)} files")

        exclude_url = "https://figshare.com/ndownloader/files/3195404"
        exclude_bytes = requests.get(exclude_url, allow_redirects=True).content
        exclude_fobj = io.TextIOWrapper(io.BytesIO(exclude_bytes))
        exclude = [int(line.split()[0]) for line in exclude_fobj.readlines()[9:-1]]
        names = [name for name in names if int(name[-10:-4]) not in exclude]
        logger.info(f"exclude files")
        QM9 = self.labels
        props = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]
        random_seleted_names = random.choices(names, k=total)

        logger.info('start to convert')
        frames = {}
        for name in random_seleted_names:
            f = io.TextIOWrapper(qm9_tar.extractfile(name))
            lines = f.readlines()
            Z = [mpot.Element[l.split()[0]].number for l in lines[2:-3]]
            R = [[float(i.replace("*^", "E")) for i in l.split()[1:4]] for l in lines[2:-3]]
            frame = mpot.Frame()
            frame[alias.Z] = torch.tensor(Z)
            frame[alias.R] = torch.tensor(R, dtype=torch.float)
            frame[alias.n_atoms] = torch.tensor([len(Z)])
            prop_line = lines[1].split()
            for k, v in zip(props, prop_line[1:]):
                frame[k] = torch.tensor([float(v)])
            frames[name] = frame
            end_time = time.perf_counter()
        logger.info('end convert')
        return frames
