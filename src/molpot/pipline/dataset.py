from functools import lru_cache
import io
import logging
import random
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import requests
import torch
from torch.nn import Module

import molpot as mpot
from molpot import NameSpace, alias, Config
import shutil

logger = logging.getLogger("molpot")


class Dataset(torch.utils.data.Dataset):
    """
    Base class for all datasets. It includes 5 processes:
        * Download / tokenize / process.
        * Clean and (maybe) save to disk.
        * Load inside Dataset.
        * Apply processs (rotate, tokenize, etc…).
        * Wrap inside a DataLoader.
    """

    labels: NameSpace

    def __init__(
        self,
        name: str,
        frames: list[mpot.Frame] = None,
        save_dir: Path | None = None,
        device: str = "cpu",
        total: int | None = None,
    ):
        super().__init__()
        self.name = name
        if save_dir is not None:
            save_dir = Path(save_dir)
            if not save_dir.exists():
                self.save_dir = Path(save_dir)
                self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir = save_dir
        self.device = device
        self.labels = NameSpace(name)
        self._preprocess = torch.nn.Sequential()

        self._processes = torch.nn.Sequential()

        self._frames = frames
        self._total = total or len(frames)

    @property
    def frames(self):
        return self._frames

    def add_process(self, process):
        self._processes.append(process)

    def apply_processs(self, frame):
        return self._processes(frame)

    def prepare_data(self):
        pass

    def __len__(self):
        return self._total

    def __iter__(self):
        return iter(map(self.apply_processs, self.frames))

    def __getitem__(self, index):
        return self.apply_processs(self._frames[index])

    def reset(self):
        self.state = {}

    def split(self, *args, **kwargs):
        return torch.utils.data.random_split(self, *args, **kwargs)

    def preprocess_data(self):
        self._frames = [self._preprocess(frame) for frame in self._frames]
        return self._frames


class QM9(Dataset):
    def __init__(
        self,
        save_dir: Path | None = None,
        device: str = "cpu",
        total: int = 133885,
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
        preprocess: list[Module] = [],
    ):

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
        frame = self.prepare_data(total)
        super().__init__("qm9", frame, save_dir, device, preprocess=preprocess)

    def __len__(self):
        return self.total

    def prepare_data(self, total):
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

        logger.info("start to convert")
        start_time = time.perf_counter()
        frames = []
        for name in random_seleted_names:
            f = io.TextIOWrapper(qm9_tar.extractfile(name))
            lines = f.readlines()
            Z = [mpot.Element[l.split()[0]].number for l in lines[2:-3]]
            R = [
                [float(i.replace("*^", "E")) for i in l.split()[1:4]]
                for l in lines[2:-3]
            ]
            frame = mpot.Frame()
            frame["global"]["name"] = name
            frame[alias.Z] = torch.tensor(Z)
            frame[alias.R] = torch.tensor(R, dtype=Config.ftype)
            frame[alias.n_atoms] = torch.tensor([len(Z)], dtype=Config.itype)
            prop_line = lines[1].split()
            for k, v in zip(props, prop_line[1:]):
                frame[k] = torch.tensor([float(v)], dtype=Config.ftype)
            frames.append(frame)
        logger.info(f"end convert, cost {time.perf_counter() - start_time:.2f}s")
        return frames


class rMD17(Dataset):

    atomrefs = {
        "energy": [
            0.0,
            -313.5150902000774,
            0.0,
            0.0,
            0.0,
            0.0,
            -23622.587180094913,
            -34219.46811826416,
            -47069.30768969713,
        ]
    }

    def __init__(
        self,
        molecule: Literal[
            "aspirin",
            "azobenzene",
            "benzene",
            "ethanol",
            "malonaldehyde",
            "naphthalene",
            "paracetamol",
            "salicylic",
            "toluene",
            "uracil",
        ],
        save_dir: str | Path,
        device: str = "cpu",
        total: int = 1000
    ):
        super().__init__(
            f"rmd17-{molecule}",
            None,
            save_dir,
            device,
            total=total,
        )
        self.labels.set("energy", "total energy", float, "kcal/mol")
        self.labels.set("forces", "all forces", float, "kcal/mol/A")

        self.molecule = molecule

        self.total = total

        self.datasets_dict = dict(
            aspirin="rmd17_aspirin.npz",
            azobenzene="rmd17_azobenzene.npz",
            benzene="rmd17_benzene.npz",
            ethanol="rmd17_ethanol.npz",
            malonaldehyde="rmd17_malonaldehyde.npz",
            naphthalene="rmd17_naphthalene.npz",
            paracetamol="rmd17_paracetamol.npz",
            salicylic_acid="rmd17_salicylic.npz",
            toluene="rmd17_toluene.npz",
            uracil="rmd17_uracil.npz",
        )

        molecules = list(self.datasets_dict.keys())
        if molecule not in molecules:
            raise ValueError(
                f"Invalid molecule. Choose from {molecules}. Got {molecule}"
            )

        self.prepare_data()
        self.preprocess_data()
        logger.info(f"Loaded {len(self)} frames")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return self.apply_processs(self.frames[idx])

    def __iter__(self):
        return iter(map(self.apply_processs, self.frames))

    @property
    def frames(self):
        return self._frames

    def prepare_data(self):

        if self.save_dir is None or not self.save_dir.exists():
            frames = self._fetch_data()
        else:
            frames = self._download_data(self.save_dir)
        self._frames = frames
        return frames

    def _fetch_data(self):
        logger.info("Fetching {} data".format(self.molecule))
        rmd17_url = "https://figshare.com/ndownloader/files/23950376"
        rmd17_bytes = requests.get(rmd17_url, allow_redirects=True).content
        rmd17_fobj = io.BytesIO(rmd17_bytes)
        rmd17_fobj.seek(0)
        rmd17_tar = tarfile.open(fileobj=rmd17_fobj, mode="r:bz2")
        data = np.load(
            rmd17_tar.extractfile(f"rmd17/npz_data/{self.datasets_dict[self.molecule]}")
        )
        return self.parse_data(data)

    def _download_data(self, save_dir: Path):

        tar_path = save_dir / "rmd17.tar.gz"
        url = "https://figshare.com/ndownloader/files/23950376"
        if not tar_path.exists():
            logger.info("Downloading {} data".format(self.molecule))
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(tar_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            logger.info("Done.")

        if not (save_dir / f"rmd17/npz_data/{self.datasets_dict[self.molecule]}").exists():
            logger.info("Extracting data...")
            tar = tarfile.open(tar_path)
            tar.extract(
                path=save_dir, member=f"rmd17/npz_data/{self.datasets_dict[self.molecule]}"
            )

        logger.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(
            save_dir / "rmd17" / "npz_data" / self.datasets_dict[self.molecule]
        )
        return self.parse_data(data)

    def parse_data(self, data):
        numbers = torch.tensor(data["nuclear_charges"], dtype=Config.itype)
        frames = []
        for positions, energies, forces in zip(
            data["coords"], data["energies"], data["forces"]
        ):
            frame = mpot.Frame()
            frame[alias.Z] = numbers
            frame[alias.R] = torch.tensor(positions, dtype=Config.ftype, requires_grad=True)
            frame["labels"]["energy"] = torch.tensor([energies - np.array(self.atomrefs['energy'])[numbers].sum()], dtype=Config.ftype)
            frame["labels"]["forces"] = torch.tensor(forces, dtype=Config.ftype)
            frame[alias.n_atoms] = torch.tensor([len(numbers)], dtype=Config.itype)
            frame[alias.cell] = torch.zeros(3)
            frame[alias.pbc] = torch.zeros(3, dtype=torch.bool)
            frames.append(frame)
            if len(frames) >= self.total:
                break
        return frames
