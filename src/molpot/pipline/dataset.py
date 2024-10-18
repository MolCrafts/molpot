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
from molpot import NameSpace, alias
import shutil

logger = logging.getLogger("molpot")


class Dataset(torch.utils.data.Dataset):
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
        save_dir: Path | None = None,
        device: str = "cpu",
        preprocess: list[Module] = [],
    ):
        super().__init__()
        self.name = name
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        self.device = device
        self.labels = NameSpace(name)
        self._preprocess = preprocess

        self.transforms = []

    def add_transform(self, transform):
        self.transforms.append(transform)

    def apply_transforms(self, frame):
        for transform in self.transforms:
            frame = transform(frame)
        return frame

    def __len__(self):
        pass

    def __iter__(self):
        pass 

    def reset(self):
        self.state = {}


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

        # self.frames = self._download_data(total)

    def __len__(self):
        return self.total


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
            frame[alias.R] = torch.tensor(R, dtype=mpot.ftype)
            frame[alias.n_atoms] = torch.tensor([len(Z)], dtype=mpot.itype)
            prop_line = lines[1].split()
            for k, v in zip(props, prop_line[1:]):
                frame[k] = torch.tensor([float(v)], dtype=mpot.ftype)
            frames.append(frame)
        logger.info(f"end convert, cost {time.perf_counter() - start_time:.2f}s")
        return frames


class rMD17(Dataset):

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
        total: int = 1000,
        preprocess: list[Module] = [],
    ):
        super().__init__("rmd17", save_dir, device, preprocess=preprocess)

        self.labels.set("energy", float, "kcal/mol", "potential_energy")
        self.labels.set("forces", torch.Tensor, "kcal/mol/A", "forces")
        self.molecule = molecule

        self.atomrefs = {
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

        # self._frames = self.prepare_data()

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return self.pre_process(self.frames[idx])

    def __iter__(self):
        return iter(map(self.pre_process, self.frames))

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
        logger.info("Downloading {} data".format(self.molecule))
        raw_path = save_dir / "rmd17"
        tar_path = save_dir / "rmd17.tar.gz"
        url = "https://figshare.com/ndownloader/files/23950376"
        if not raw_path.exists():
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(tar_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
        logger.info("Done.")

        logger.info("Extracting data...")
        tar = tarfile.open(tar_path)
        tar.extract(
            path=raw_path, member=f"rmd17/npz_data/{self.datasets_dict[self.molecule]}"
        )

        logger.info("Parsing molecule {:s}".format(self.molecule))

        data = np.load(
            raw_path / "rmd17" / "npz_data" / self.datasets_dict[self.molecule]
        )
        return self.parse_data(data)

    def parse_data(self, data):
        numbers = data["nuclear_charges"]
        frames = []
        for positions, energies, forces in zip(
            data["coords"], data["energies"], data["forces"]
        ):
            frame = mpot.Frame()
            frame[alias.Z] = torch.tensor(numbers, dtype=mpot.itype)
            frame[alias.R] = torch.tensor(positions, dtype=mpot.ftype)
            frame[alias.props_ns.name]["energy"] = torch.tensor(
                [energies], dtype=mpot.ftype
            )
            frame[alias.props_ns.name]["forces"] = torch.tensor(
                forces, dtype=mpot.ftype
            )
            frame[alias.n_atoms] = torch.tensor([len(numbers)], dtype=mpot.itype)
            frame[alias.cell] = torch.zeros(3)
            frame[alias.pbc] = torch.zeros(3, dtype=torch.bool)
            frames.append(frame)

        return frames
