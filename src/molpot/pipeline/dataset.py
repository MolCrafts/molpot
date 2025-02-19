import io
import logging
import random
import tarfile, bz2
import time
from pathlib import Path
from typing import Literal, Sequence, Any

import numpy as np
import requests
import torch
from torch.nn import Module, Sequential

import molpot as mpot
from molpot import Config, NameSpace, alias

from abc import abstractmethod

from molpot.process import AtomicDress

from tqdm import tqdm

logger = logging.getLogger("molpot")


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name: str,
        save_dir: Path | None = None,
        device: str = "cpu",
        # processes: list[Module] = [],
    ):
        self.labels: NameSpace = NameSpace(name)
        self.device = device
        # self._data_process = Sequential(*processes)

        if save_dir is not None:
            self.save_dir = Path(save_dir)
            if not self.save_dir.exists():  # create save_dir
                self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

    def prepare(self): ...

    def download(self): ...


class IterStyleDataset(Dataset):

    def __init__(self, frames: Sequence[mpot.Frame]):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i: int) -> Any:
        return self.frames[i]


class MapStyleDataset(Dataset):

    @abstractmethod
    def __getitem__(self, i: int) -> Any: ...

    @abstractmethod
    def __len__(self): ...


class QM9(Dataset):
    def __init__(
        self,
        save_dir: Path | None = None,
        device: str = "cpu",
        atom_ref: bool = True,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("qm9", save_dir=save_dir, device=device)

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

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    def prepare(self, total: int = None, preprocess=[]):

        logger.info("downloading...")

        def get_content(save_dir: Path | None):
            if save_dir and (save_dir / "qm9.tar.bz2").exists():
                qm9_bytes = io.BytesIO((save_dir / "qm9.tar.bz2").read_bytes())
                exclude_txt = (save_dir / "exclude.txt").read_text()
                return qm9_bytes, exclude_txt

            qm9_url = "https://ndownloader.figshare.com/files/3195389"
            exclude_url = "https://figshare.com/ndownloader/files/3195404"
            qm9_bytes = requests.get(qm9_url, allow_redirects=True).content
            exclude_bytes = requests.get(exclude_url, allow_redirects=True).content
            exclude_txt = exclude_bytes.decode("utf-8")
            if save_dir:
                with open(save_dir / "qm9.tar.bz2", "wb") as f:
                    f.write(qm9_bytes)
                with open(save_dir / "exclude.txt", "w") as f:
                    f.write(exclude_txt)
            # stream
            return qm9_bytes, exclude_txt

        qm9_bytes, exclude_txt = get_content(self.save_dir)
        logger.info("done")
        exclude = set(int(line.split()[0]) for line in exclude_txt.split("\n")[9:-2])

        start_extract_time = time.perf_counter()

        with tarfile.open(fileobj=qm9_bytes, mode="r:bz2") as tar_file:
            names = set(tar_file.getnames())
            exclude = set(exclude)
            names = names - exclude
            if total is None:
                total = len(names)
            random_seleted_names = random.sample(list(names), k=total)
            end_extract_time = time.perf_counter()
            logger.info(
                f"end extract, cost {end_extract_time - start_extract_time:.2f}s, average {(end_extract_time - start_extract_time)/total:.2f} s/file"
            )

            QM9 = self.labels
            props = [QM9.zpve, QM9.U0, QM9.U, QM9.H, QM9.G, QM9.Cv]
            logger.info("parsing...")
            start_time = time.perf_counter()
            frames = []

            # extract to tmp dir
            import tempfile

            tmp_dir = Path(tempfile.gettempdir()) / "qm9_extracted"
            tmp_dir.mkdir(exist_ok=True)
            tar_file.extractall(tmp_dir)
            for name in tqdm(random_seleted_names):
                with open(tmp_dir / name, "r") as f:
                    lines = f.readlines()
                    n_atoms = int(lines[0])
                    Z = [mpot.Element(l.split()[0]).number for l in lines[2:-3]]
                    R = [
                        [float(i.replace("*^", "E")) for i in l.split()[1:4]]
                        for l in lines[2:-3]
                    ]
                    frame = mpot.Frame()
                    frame[alias.Z] = torch.tensor(Z)
                    frame[alias.R] = torch.tensor(R, dtype=Config.ftype)
                    frame[alias.n_atoms] = torch.tensor(n_atoms, dtype=Config.itype)
                    prop_line = lines[1].split()
                    for k, v in zip(props, prop_line[1:]):
                        frame["labels", k[-1]] = torch.tensor(
                            [float(v)], dtype=Config.ftype
                        )
                    frames.append(frame)
            ## === on the fly parsing ===
            # files = (
            #     tar_file.extractfile(name).read().decode('utf-8').split('\n') for name in random_seleted_names
            # )  # use a generator to lazy load
            # for lines in tqdm(files):
            #     n_atoms = int(lines[0])
            #     Z = [mpot.Element(l.split()[0]).number for l in lines[2:-4]]
            #     R = [
            #         [float(i.replace("*^", "E")) for i in l.split()[1:4]]
            #         for l in lines[2:-4]
            #     ]
            #     frame = mpot.Frame()
            #     # frame["props", "name"] = lines.stem
            #     frame[alias.Z] = torch.tensor(Z)
            #     frame[alias.R] = torch.tensor(R, dtype=Config.ftype)
            #     frame[alias.n_atoms] = torch.tensor(n_atoms, dtype=Config.itype)
            #     prop_line = lines[1].split()
            #     for k, v in zip(props, prop_line[1:]):
            #         frame["labels", k[-1]] = torch.tensor(
            #             [float(v)], dtype=Config.ftype
            #         )
            #     frames.append(frame)
        logger.info(f"end parse, cost {time.perf_counter() - start_time:.2f}s")

        atomic_dress = AtomicDress(dress_key=("labels", "U0"))

        _preprocess = Sequential(*preprocess)
        self._frames = [_preprocess(frame) for frame in frames]
        self._frames = atomic_dress(self._frames)

        self._frames = frames
        return frames


class rMD17(MapStyleDataset):

    atomrefs = {
        "energy": torch.tensor(
            [
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
        )
    }

    datasets_dict = dict(
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
        save_dir: Path | None = None,
        device: str = "cpu",
        atomic_dress: bool = True,
    ):
        super().__init__(
            name="rmd17",
            save_dir=save_dir,
            device=device,
        )
        self.labels.set(
            "energy", "total energy", float, "kcal/mol", (None, 1), "labels"
        )
        self.labels.set(
            "forces", "all forces", float, "kcal/mol/A", (None, 3), "labels"
        )

        self.molecule = molecule

        # self.total = total if total is not None else torch.inf

        molecules = list(self.datasets_dict.keys())
        if molecule not in molecules:
            raise ValueError(
                f"Invalid molecule. Choose from {molecules}. Got {molecule}"
            )

        self.archive_path = self.save_dir / "rmd17.tar.gz"
        self.npz_path = (
            self.save_dir / self.molecule / self.datasets_dict[self.molecule]
        )  # save_dir/aspirin/rmd17_aspirin.npz

        self._frames = []

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]

    @property
    def frames(self):
        return self._frames

    def prepare(self, total: int | None = 1000, preprocess: list[Module] = []):

        if self.save_dir is None:
            data = self._fetch_data()
        else:
            data = self.download()
        raw_frames = self.parse_data(data, total=total)

        atomic_dress = AtomicDress()

        _preprocess = Sequential(*preprocess)
        self._frames = [_preprocess(frame) for frame in raw_frames]
        self._frames = atomic_dress(self._frames)
        return self._frames

    def _fetch_data(self) -> Sequence[mpot.Frame]:
        logger.info(f"Fetching {self.molecule} data")
        rmd17_url = "https://figshare.com/ndownloader/files/23950376"
        rmd17_bytes = requests.get(rmd17_url, allow_redirects=True).content
        rmd17_fobj = io.BytesIO(rmd17_bytes)
        rmd17_fobj.seek(0)
        rmd17_tar = tarfile.open(fileobj=rmd17_fobj, mode="r:bz2")
        data = np.load(rmd17_tar.extractfile(self.npz_path))
        return data

    def download(self) -> Sequence[mpot.Frame]:

        url = "https://figshare.com/ndownloader/files/23950376"
        if not self.archive_path.exists():
            logger.info(f"Downloading {self.molecule} data")
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(self.archive_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
            logger.info("Done.")

        if not self.npz_path.exists():
            logger.info("Extracting data...")
            with tarfile.open(self.archive_path) as tar:
                tar.extract(
                    path=self.npz_path.parent,
                    member=f"rmd17/npz_data/{self.datasets_dict[self.molecule]}",
                    filter=lambda member, path: member.replace(
                        name=self.datasets_dict[self.molecule]
                    ),
                )

        logger.info(f"Parsing molecule {self.molecule}")

        data = np.load(self.npz_path)
        return data

    def parse_data(self, data, total: int) -> Sequence[mpot.Frame]:
        numbers = torch.tensor(data["nuclear_charges"], dtype=Config.itype)
        frames = []
        indices = torch.randperm(len(data["coords"]))[:total]
        for idx in indices:
            frame = mpot.Frame()
            frame[alias.Z] = numbers
            frame[alias.R] = torch.tensor(data["coords"][idx], dtype=torch.float32)
            frame["labels", "energy"] = torch.tensor(
                [
                    data["energies"][idx]
                ],  # - torch.sum(self.atomrefs["energy"][numbers])
                dtype=Config.ftype,
            )
            frame["labels", "forces"] = torch.tensor(
                data["forces"][idx], dtype=Config.ftype
            )
            frame[alias.n_atoms] = torch.tensor([len(numbers)], dtype=Config.itype)
            frames.append(frame)
        return frames

    def save(self, *args, **kwargs):
        frames = mpot.Frame.from_frames(self._frames)
        frames.save(self.save_dir / f"{self.molecule}.pt", *args, **kwargs)
        return frames

    def load(self, *args, **kwargs):
        frames = mpot.Frame.load(self.save_dir / f"{self.molecule}.pt", *args, **kwargs)
        self._frames = [mpot.Frame(frame) for frame in frames]
        return self._frames
