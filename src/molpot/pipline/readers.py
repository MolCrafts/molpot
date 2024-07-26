import logging
import tarfile as tar
from typing import Iterable

import numpy as np
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

import molpy as mp
import molpot as mpot

from molpot.alias import *
from molpot.config import Config

# @functional_datapipe("read_chemfiles")
# class ChemFilesReader(IterDataPipe):
#     def __init__(
#         self, source_dp: IterDataPipe, ext: str, in_memory: bool = True, **kwargs
#     ):
#         super().__init__()
#         self.source_dp = source_dp
#         self.kwargs = kwargs
#         self.ext = ext
#         self.in_memory = in_memory

#     def __iter__(self) -> mp.Frame:
#         if self.in_memory:
#             for d in self.source_dp:
#                 loader = mp.MemoryLoader(d)
#                 yield loader.load_frame()

#         else:
#             for d in self.source_dp:
#                 loader = mp.DataLoader(d)
#                 yield loader.load_frame()


@functional_datapipe("read_qm9")
class QM9Reader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self) -> Iterable[dict]:

        qm9_ns = mpot.NameSpace("QM9")
        qm9_keys = [alias.key for alias in qm9_ns.values()]
        for path, stream in self.source_dp:
            lines = stream.decode().split('\n')
            frame = dict()
            frame[n_atoms] = torch.tensor(int(lines[0]), dtype=Config.stype)
            props_line = lines[1].split()[1:]
            frame[atomid] = torch.tensor(int(props_line[0]), dtype=Config.stype)
            for prop, p in zip(qm9_keys, props_line[1:]):
                value = torch.tensor(float(p))
                frame[prop] = torch.atleast_1d(value).to(Config.device).to(Config.ftype)

            frame[xyz] = torch.tensor(
                [
                    [float(i.replace("*^", "E")) for i in line.split()[1:4]]
                    for line in lines[2:2+frame[n_atoms].item()]
                ], device=mpot.Config.device
            )

            frame[Z] = torch.tensor(
                [
                    mp.Element[line.split()[0]].number
                    for line in lines[2:2+frame[n_atoms].item()]
                ],
                dtype=Config.stype,
                device=Config.device,
            )

            frame[cell] = torch.zeros((3, 3), device=Config.device)
            frame[pbc] = torch.tensor([False, False, False], device=Config.device)

            yield frame


# @functional_datapipe("read_rmd17")
# class rMD17Reader(IterDataPipe):
#     def __init__(self, source_dp: IterDataPipe):
#         super().__init__()
#         self.source_dp = source_dp

#     def __iter__(self):
#         for data in self.source_dp:
#             data = np.load(data, allow_pickle=True)

#             numbers = data["nuclear_charges"]
#             for positions, energies, forces in zip(
#                 data["coords"], data["energies"], data["forces"]
#             ):
#                 frame = {}
#                 frame[Alias.n_atoms] = torch.tensor(len(numbers), dtype=Config.stype)
#                 frame[Alias.rmd17.energy] = torch.tensor([energies], dtype=Config.ftype)
#                 frame[Alias.rmd17.forces] = torch.tensor(forces, dtype=Config.ftype)
#                 frame[Alias.Z] = torch.tensor(numbers, dtype=Config.stype)
#                 frame[Alias.R] = torch.tensor(positions, dtype=Config.ftype)
#                 frame[Alias.cell] = torch.zeros((3, 3), dtype=Config.ftype)
#                 frame[Alias.pbc] = torch.tensor([False, False, False])

#                 yield frame
