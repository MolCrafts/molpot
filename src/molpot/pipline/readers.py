from typing import Iterable

import numpy as np
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

import molpot as mpot

from molpot import alias

@functional_datapipe("read_qm9")
class QM9Reader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self) -> Iterable[dict]:

        qm9_ns = mpot.NameSpace("qm9")
        qm9_keys = [alias.key for alias in qm9_ns.values()]
        for path, stream in self.source_dp:
            lines = stream.decode().split('\n')
            frame = mpot.Frame()
            _n_atoms = int(lines[0])
            frame[alias.n_atoms] = torch.tensor(_n_atoms)
            props_line = lines[1].split()[1:]
            frame[alias.atomid] = torch.tensor(int(props_line[0]))
            for prop, p in zip(qm9_keys, props_line[1:]):
                value = torch.tensor(float(p))
                frame[prop] = torch.atleast_1d(value)

            frame[alias.xyz] = torch.tensor(
                [
                    [float(i.replace("*^", "E")) for i in line.split()[1:4]]
                    for line in lines[2:2+_n_atoms]
                ], device=mpot.Config.device
            )

            frame[alias.Z] = torch.tensor(
                [
                    mpot.Element[line.split()[0]].number
                    for line in lines[2:2+_n_atoms]
                ]
            )

            frame[alias.box_matrix] = torch.zeros((3, 3))
            frame[alias.pbc] = torch.tensor([False, False, False])
            yield frame
