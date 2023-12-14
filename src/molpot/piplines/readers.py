from torchdata.datapipes import functional_datapipe
import molpot as mpot
import molpy as mp
from typing import Iterable
from torchdata.datapipes.iter import IterDataPipe
from molpot import alias
import torch

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
            frame = dict()
            lines = d[1].readlines()
            frame[alias.natoms] = torch.tensor(int(lines[0]), dtype=torch.int32)
            props_line = lines[1].split()[1:]
            frame[alias.idx] = torch.tensor(int(props_line[0]), dtype=torch.int32)
            for prop, p in zip(alias.QM9.values(), props_line[1:]):
                if prop in alias:
                    src_unit = prop.unit
                    dst_unit = alias.get_unit(prop)
                    frame[prop.key] = mp.units.convert(float(p), src_unit, dst_unit)
                else:
                    frame[prop.key] = torch.tensor(float(p))

            xyz = torch.tensor(
                [
                    [float(i.replace("*^", "E")) for i in line.split()[1:4]]
                    for line in lines[2:-3]
                ]
            )

            frame[alias.xyz] = mp.units.convert(xyz, "angstrom", alias["xyz"].unit)

            frame[alias.Z] = torch.tensor([
                mp.Element.get_atomic_number_by_symbol(line.split()[0])
                for line in lines[2:-3]
            ], dtype=torch.int32)

            frame[alias.cell] = torch.zeros((3, 3))
            frame[alias.pbc] = torch.tensor([False, False, False])

            yield frame
