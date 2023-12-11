from torchdata.datapipes import functional_datapipe
import molpot as mpot
import molpy as mp
from typing import Iterable
from torchdata.datapipes.iter import IterDataPipe
from molpot import alias

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

    def __iter__(self) -> Iterable[mp.Frame]:
        
        for d in self.source_dp:
            frame = mp.Frame()
            lines = d[1].readlines()
            frame[alias.natoms] = int(lines[0])
            props_line = lines[1].split()[1:]
            frame[alias.idx] = int(props_line[0])
            for prop, p in zip(alias.alias(), props_line[1:]):
                if prop in alias:
                    src_unit = alias.QM9.get_unit(prop)
                    dst_unit = alias.get_unit(prop)
                    frame[prop] = mp.units.convert(float(p), src_unit, dst_unit)
                else:
                    frame[prop] = float(p)

            frame.atoms[alias.xyz] = [
                [i.replace("*^", "E") for i in line.split()[1:4]]
                for line in lines[2:-3]
            ]
            frame.atoms[alias.Z] = [
                mp.Element.get_atomic_number_by_symbol(line.split()[0])
                for line in lines[2:-3]
            ]

            yield frame
