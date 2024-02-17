from molpot.piplines.dataset import rMD17
from torchdata.datapipes import functional_datapipe
import molpot as mpot
import molpy as mp
from typing import Iterable
from torchdata.datapipes.iter import IterDataPipe
from molpot import alias, Config
import torch
import logging
import numpy as np
import tarfile as tar

__all__ = [
    "ChemFilesReader",
    "QM9Reader",
]


@functional_datapipe("read_chemfiles")
class ChemFilesReader(IterDataPipe):
    def __init__(
        self, source_dp: IterDataPipe, ext: str, in_memory: bool = True, **kwargs
    ):
        super().__init__()
        self.source_dp = source_dp
        self.kwargs = kwargs
        self.ext = ext
        self.in_memory = in_memory

    def __iter__(self) -> mp.Frame:
        if self.in_memory:
            for d in self.source_dp:
                loader = mp.MemoryLoader(d)
                yield loader.load_frame()

        else:
            for d in self.source_dp:
                loader = mp.DataLoader(d)
                yield loader.load_frame()


@functional_datapipe("read_qm9")
class QM9Reader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self) -> Iterable[dict]:
        for d in self.source_dp:
            print(d)
            frame = dict()
            lines = d[1].readlines()
            frame[alias.natoms] = torch.tensor(int(lines[0]), dtype=Config.stype)
            props_line = lines[1].split()[1:]
            frame[alias.idx] = torch.tensor(int(props_line[0]), dtype=Config.stype)
            for prop, p in zip(alias.QM9.items(), props_line[1:]):
                if prop in alias:
                    src_unit = prop.unit
                    dst_unit = alias.get_unit(prop)
                    value = mp.units.convert(float(p), src_unit, dst_unit)
                else:
                    value = torch.tensor(float(p))
                frame[prop.key] = torch.atleast_1d(value).to(Config.device).to(Config.ftype)

            xyz = torch.tensor(
                [
                    [float(i.replace("*^", "E")) for i in line.split()[1:4]]
                    for line in lines[2:-3]
                ], device=Config.device
            )

            frame[alias.xyz] = mp.units.convert(xyz, "angstrom", alias["xyz"].unit)

            frame[alias.Z] = torch.tensor(
                [
                    mp.Element.get_atomic_number_by_symbol(line.split()[0])
                    for line in lines[2:-3]
                ],
                dtype=Config.stype,
                device=Config.device,
            )

            frame[alias.cell] = torch.zeros((3, 3), device=Config.device)
            frame[alias.pbc] = torch.tensor([False, False, False], device=Config.device)

            yield frame


@functional_datapipe("read_rmd17")
class rMD17Reader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self) -> mp.Frame:
        for dp in self.source_dp:
            data = np.load(dp)

            numbers = data["nuclear_charges"]
            for positions, energies, forces in zip(
                data["coords"], data["energies"], data["forces"]
            ):
                frame = {}
                frame[alias.natoms] = torch.tensor(len(numbers), dtype=Config.stype)
                frame[alias.energy] = torch.tensor([energies]) * 43.3641
                frame[alias.forces] = torch.tensor(forces) * 43.3641
                frame[alias.Z] = torch.tensor(numbers)
                frame[alias.R] = mp.units.convert(
                    torch.tensor(positions), alias.rMD17["R"].unit, alias["R"].unit
                )
                frame[alias.cell] = torch.zeros((3, 3))
                frame[alias.pbc] = torch.tensor([False, False, False])
                yield frame

            logging.info("Done.")
