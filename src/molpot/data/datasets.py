from functools import partial
import torch
from torchdata.datapipes.iter import IterDataPipe, FileLister, IterableWrapper
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
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
from molpot import alias
import molpy as mp

__all__ = ["DataSet", "DataLoader2", "QM9"]


log = logging.getLogger(__name__)


def endswith(x: str, suffix):
    return x.endswith(suffix)


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
        name,
        data_dir: Optional[Path | str],
        pipelines: list[dict] = {},
        num_workers: int = 0,
    ):
        super().__init__()
        self.name = name
        self._pipelines = pipelines
        if data_dir is None:
            self.data_dir = Path(tempfile.mkdtemp())
        else:
            self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers

    def update_meta(self, data: Optional[dict[str, Any]] = None):
        _data = {
            "name": self.name,
            "update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        _data.update(data or {})
        with open(self.data_dir / Path("meta.json"), "w") as f:
            json.dump(_data, f)
        self.is_prepared = True

    def set_keywords(self, aliases: dict[str, str]):
        self.aliases = aliases

    def prepare(self):
        raise NotImplementedError

    def _create_dataloader(self, datapipe: IterDataPipe) -> DataLoader2:
        for pipeline in self._pipelines:
            args = pipeline.get("args", {})
            datapipe = getattr(datapipe, pipeline['type'])(**args)
            if isinstance(datapipe, tuple):
                break

        if self.num_workers:
            rs = MultiProcessingReadingService(self.num_workers)
        else:
            rs = None
        make_dataloader = partial(DataLoader2, reading_service=rs)
        if isinstance(datapipe, tuple):
            return list(map(make_dataloader, datapipe))
        else:
            return make_dataloader(datapipe)

    def fetch(self, url, filename, dir: Optional[Path | str] = None) -> Path:
        if dir is None:
            dir = self.data_dir

        fpath = Path(dir) / Path(filename)
        if not fpath.exists():
            log.info(f"downloading from {url}... to {fpath}")
            urlretrieve(url, fpath)
        else:
            log.info(f"{fpath} already exists.")
        return fpath


class QM9(DataSet):
    def __init__(
        self,
        data_dir: Optional[Path | str] = None,
        pipelines: dict[str, dict[str, Any]] = {},
        num_workers: int = 0,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("QM9", data_dir, pipelines, num_workers)
        self.remove_uncharacterized = remove_uncharacterized
        self.aliases = mpot.Aliases("QM9")
        self.aliases.set("A", "rotational_constant_A", "GHz", "")
        self.aliases.set("B", "rotational_constant_B", "GHz", "")
        self.aliases.set("C", "rotational_constant_C", "GHz", "")
        self.aliases.set("mu", "dipole_moment", "Debye", "")
        self.aliases.set("alpha", "isotropic_polarizability", "a0 a0 a0", "")
        self.aliases.set("homo", "homo", "Ha", "")
        self.aliases.set("lumo", "lumo", "Ha", "")
        self.aliases.set("gap", "gap", "Ha", "")
        self.aliases.set("r2", "electronic_spatial_extent", "a0 a0", "")
        self.aliases.set("zpve", "zpve", "Ha", "")
        self.aliases.set("U0", "energy_U0", "", "")
        self.aliases.set("U", "energy_U", "Ha", "")
        self.aliases.set("H", "enthalpy_H", "Ha", "")
        self.aliases.set("G", "free_energy", "Ha", "")
        self.aliases.set("Cv", "heat_capacity", "cal/mol/K", "")

    def prepare(self) -> DataLoader2:
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
            IterableWrapper((map(str, np.array(ordered_files)[irange])))
            # can not use lambda due to it is not pickleable
            .filter(filter_fn=partial(endswith, suffix=".xyz"))
            .open_files(mode="rt")
            .read_xyz(aliases=self.aliases)
        )
        return super()._create_dataloader(dp)

    def _download_atomrefs(self):
        url = "https://ndownloader.figshare.com/files/3195395"
        filename = "atomref.txt"
        atomrefs_path = self.fetch(url, filename)
        props = [
            self.aliases.zpve,
            self.aliases.U0,
            self.aliases.U,
            self.aliases.H,
            self.aliases.G,
            self.aliases.Cv,
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
        path = self.fetch(at_url, "uncharacterized.txt")
        uncharacterized = []
        with open(path) as f:
            lines = f.readlines()
            for line in lines[9:-1]:
                uncharacterized.append(int(line.split()[0]))
        return uncharacterized

    def _download_data(self):
        url = "https://ndownloader.figshare.com/files/3195389"
        tar_path = self.fetch(url, "gdb9.tar.gz")
        raw_path = self.data_dir / Path("gdb9_xyz")

        if not raw_path.exists():
            log.info("Extracting files...")
            tar = tarfile.open(tar_path)
            tar.extractall(raw_path)
            tar.close()
            log.info("Done.")

        log.info("Parse xyz files...")
        ordered_files = sorted(
            raw_path.rglob("*.xyz"),
            key=lambda x: (int(re.sub(r"\D", "", str(x))), str(x)),
        )  # sort by index in filename
        return ordered_files


class _GDMLDataModule(DataSet):
    def __init__(
        self,
        molecule: str,
        datasets_dict: dict[str, str],
        download_url: str,
        data_dir: Optional[Path | str],
    ):
        super().__init__("_GDMLData", data_dir)
        self.url = download_url
        self.datasets_dict = datasets_dict
        self.molecule = molecule

    def prepare(self) -> DataLoader2:
        properties = self._download_data()
        dp = IterableWrapper(properties).in_memory_cache()

        # rs = MultiProcessingReadingService(num_workers=1)
        dl = DataLoader2(dp)
        return dl

    def _download_data(
        self,
    ):
        raw_path = self.fetch(self.url, self.datasets_dict[self.molecule])
        data = np.load(raw_path)

        numbers = data["z"]
        frames = []
        for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
            frame = mp.Frame()
            frame[alias.energy] = (
                energies if type(energies) is np.ndarray else np.array([energies])
            )
            frame[alias.forces] = forces
            frame[alias.Z] = numbers
            frame[alias.R] = positions
            frame.box.set_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
            frame.box.pbc = np.array([False, False, False])
            frames.append(frames)

        return frames


class MD17(_GDMLDataModule):
    atomrefs = {
        alias.energy: [
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

    datasets_dict = dict(
        aspirin="md17_aspirin.npz",
        # aspirin_ccsd='aspirin_ccsd.zip',
        azobenzene="azobenzene_dft.npz",
        benzene="md17_benzene2017.npz",
        ethanol="md17_ethanol.npz",
        # ethanol_ccsdt='ethanol_ccsd_t.zip',
        malonaldehyde="md17_malonaldehyde.npz",
        # malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
        naphthalene="md17_naphthalene.npz",
        paracetamol="paracetamol_dft.npz",
        salicylic_acid="md17_salicylic.npz",
        toluene="md17_toluene.npz",
        # toluene_ccsdt='toluene_ccsd_t.zip',
        uracil="md17_uracil.npz",
    )

    def __init__(self, data_dir: Optional[Path | str], molecule: str):
        super().__init__(
            molecule,
            self.datasets_dict,
            "http://www.quantum-machine.org/gdml/data/npz/",
            data_dir,
        )


class MD22(_GDMLDataModule):
    atomrefs = {
        alias.energy: [
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
    datasets_dict = {
        "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
        "DHA": "md22_DHA.npz",
        "stachyose": "md22_stachyose.npz",
        "AT-AT": "md22_AT-AT.npz",
        "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
        "buckyball-catcher": "md22_buckyball-catcher.npz",
        "double-walled_nanotube": "md22_double-walled_nanotube.npz",
    }

    def __init__(self, data_dir: Optional[Path | str], molecule: str):
        super().__init__(
            molecule,
            self.datasets_dict,
            "http://www.quantum-machine.org/gdml/repo/datasets/",
            data_dir,
        )
