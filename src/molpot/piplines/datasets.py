from functools import partial
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchdata.dataloader2 import DataLoader2
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
import molpy as mp

__all__ = ["DataSet", "DataLoader2", "QM9"]


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
        self,
        name,
        data_dir: None | Path | str,
        in_memory: bool = True,
    ):
        super().__init__()
        self.name = name
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(tempfile.mkdtemp())
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

        self.in_memory = in_memory

    def update_meta(self, data: Optional[dict[str, Any]] = None):
        self._data = {
            "name": self.name,
            "update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        self._data.update(data or {})
        if not self.in_memory:
            with open(self.data_dir / Path("meta.json"), "w") as f:
                json.dump(self._data, f)

    def prepare(self) -> IterDataPipe:
        raise NotImplementedError

    def fetch(self, url, filename, dir: Path | str) -> Path:
        fpath = Path(dir) / Path(filename)
        if dir is None:
            dir = self.data_dir
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
        in_memory: bool = False,
        remove_uncharacterized: bool = True,
    ):
        super().__init__("QM9", data_dir, in_memory)
        self.remove_uncharacterized = remove_uncharacterized
        mpot.alias("QM9")
        mpot.alias.QM9.set("A", "_A", float, "GHz", "rotational_constant_A")
        mpot.alias.QM9.set("B", "_B", float, "GHz", "rotational_constant_B")
        mpot.alias.QM9.set("C", "_C", float, "GHz", "rotational_constant_C")
        mpot.alias.QM9.set("mu", "_mu", float, "Debye", "dipole_moment")
        mpot.alias.QM9.set("alpha", "_alpha", float, "a0 a0 a0", "isotropic_polarizability")
        mpot.alias.QM9.set("homo", "_homo", float, "Ha", "homo")
        mpot.alias.QM9.set("lumo", "_lumo", float, "Ha", "lump")
        mpot.alias.QM9.set("gap", "_gap", float, "Ha", "gap")
        mpot.alias.QM9.set("r2", "_r2", float, "a0 a0", "electronic_spatial_extent")
        mpot.alias.QM9.set("zpve", "_zpve", float, "Ha", "zpve")
        mpot.alias.QM9.set("U0", "_U0", float, "", "_energy_U0")
        mpot.alias.QM9.set("U", "_U", float, "Ha", "_energy_U")
        mpot.alias.QM9.set("H", "_H", float, "Ha", "_enthalpy_H")
        mpot.alias.QM9.set("G", "_G", float, "Ha", "_free_energy")
        mpot.alias.QM9.set("Cv", "_Cv", float, "cal/mol/K", "_heat_capacity")

    def prepare(self) -> IterDataPipe:
        if self.in_memory:
            import requests
            import io

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
            # dp = IterableWrapper(names).map(
            #     lambda x: io.TestIOWrapper(
            #         qm9_tar.extractfile(x)
            #     ).readlines()
            # ).read_qm9()

        else:
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
                IterableWrapper(map(str, np.array(ordered_files)[irange]))
                .read_qm9()
            )
        return dp

    def _download_atomrefs(self):
        url = "https://ndownloader.figshare.com/files/3195395"
        filename = "atomref.txt"
        atomrefs_path = self.fetch(url, filename, self.data_dir)
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
        raw_path = self.data_dir / Path("gdb9_xyz")

        _sort_qm9 = lambda x: (int(re.sub(r"\D", "", str(x))), str(x))
        if not raw_path.exists():
            log.info("Extracting files...")
            tar = tarfile.open(tar_path)
            tar.extractall(raw_path)
            tar.close()
            log.info("Done.")

        log.info("Parse xyz files...")
        ordered_files = sorted(
            raw_path.rglob("*.xyz"),
            key=_sort_qm9,
        )  # sort by index in filename
        return ordered_files


# class _GDMLDataModule(DataSet):
#     def __init__(
#         self,
#         molecule: str,
#         datasets_dict: dict[str, str],
#         download_url: str,
#         data_dir: Optional[Path | str],
#     ):
#         super().__init__("_GDMLData", data_dir)
#         self.url = download_url
#         self.datasets_dict = datasets_dict
#         self.molecule = molecule

#     def prepare(self) -> DataLoader2:
#         properties = self._download_data()
#         dp = IterableWrapper(properties).in_memory_cache()

#         # rs = MultiProcessingReadingService(num_workers=1)
#         dl = DataLoader2(dp)
#         return dl

#     def _download_data(
#         self,
#     ):
#         raw_path = self.fetch(self.url, self.datasets_dict[self.molecule])
#         data = np.load(raw_path)

#         numbers = data["z"]
#         frames = []
#         for positions, energies, forces in zip(data["R"], data["E"], data["F"]):
#             frame = mp.Frame()
#             frame[mpot.Alias.energy] = (
#                 energies if type(energies) is np.ndarray else np.array([energies])
#             )
#             frame[mpot.Alias.forces] = forces
#             frame[mpot.Alias.Z] = numbers
#             frame[mpot.Alias.R] = positions
#             frame.box.set_matrix(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
#             frame.box.pbc = np.array([False, False, False])
#             frames.append(frames)

#         return frames


# class MD17(_GDMLDataModule):
#     atomrefs = {
#         mpot.Alias.energy: [
#             0.0,
#             -313.5150902000774,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             -23622.587180094913,
#             -34219.46811826416,
#             -47069.30768969713,
#         ]
#     }

#     datasets_dict = dict(
#         aspirin="md17_aspirin.npz",
#         # aspirin_ccsd='aspirin_ccsd.zip',
#         azobenzene="azobenzene_dft.npz",
#         benzene="md17_benzene2017.npz",
#         ethanol="md17_ethanol.npz",
#         # ethanol_ccsdt='ethanol_ccsd_t.zip',
#         malonaldehyde="md17_malonaldehyde.npz",
#         # malonaldehyde_ccsdt='malonaldehyde_ccsd_t.zip',
#         naphthalene="md17_naphthalene.npz",
#         paracetamol="paracetamol_dft.npz",
#         salicylic_acid="md17_salicylic.npz",
#         toluene="md17_toluene.npz",
#         # toluene_ccsdt='toluene_ccsd_t.zip',
#         uracil="md17_uracil.npz",
#     )

#     def __init__(self, data_dir: Optional[Path | str], molecule: str):
#         super().__init__(
#             molecule,
#             self.datasets_dict,
#             "http://www.quantum-machine.org/gdml/data/npz/",
#             data_dir,
#         )


# class MD22(_GDMLDataModule):
#     atomrefs = {
#         mpot.Alias.energy: [
#             0.0,
#             -313.5150902000774,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             -23622.587180094913,
#             -34219.46811826416,
#             -47069.30768969713,
#         ]
#     }
#     datasets_dict = {
#         "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",
#         "DHA": "md22_DHA.npz",
#         "stachyose": "md22_stachyose.npz",
#         "AT-AT": "md22_AT-AT.npz",
#         "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
#         "buckyball-catcher": "md22_buckyball-catcher.npz",
#         "double-walled_nanotube": "md22_double-walled_nanotube.npz",
#     }

#     def __init__(self, data_dir: Optional[Path | str], molecule: str):
#         super().__init__(
#             molecule,
#             self.datasets_dict,
#             "http://www.quantum-machine.org/gdml/repo/datasets/",
#             data_dir,
#         )
