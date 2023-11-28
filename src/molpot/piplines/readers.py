from typing import Sequence
from torchdata.datapipes.iter import IterDataPipe, Collator
from torchdata.datapipes import functional_datapipe
import numpy as np
import molpy as mp
import molpot as mpot
import torch
from molpot import alias, Aliases
from molpot.transforms import TorchNeighborList
from typing import Iterable, Optional

__all__ = [
    "XYZReader",
    "CollateFrames",
    "CalcNBList",
]


@functional_datapipe("read_xyz")
class XYZReader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        self.source_dp = source_dp
        self.local_alias = alias

    def __iter__(self) -> Iterable[mp.Frame]:
        local_alias = self.local_alias

        for d in self.source_dp:
            frame = mp.Frame()

            lines = d[1].readlines()
            frame[alias.natoms] = int(lines[0])
            props_line = lines[1].split()[1:]
            frame["index"] = int(props_line[0])
            for prop, p in zip(local_alias.get_aliases(), props_line[1:]):
                if prop in local_alias:
                    src_unit = local_alias.get_unit(prop)
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


def _collate(batch: Sequence[mp.Frame]):
    assert isinstance(batch, Iterable), "batch must be an iterable"

    coll_batch = {}

    props_keys = batch[0]._props.keys()
    atoms_keys = batch[0].atoms.keys()

    for k in props_keys:
        coll_batch[k] = torch.concat(
            [torch.atleast_1d(torch.tensor(frame[k])) for frame in batch]
        )

    for k in atoms_keys:
        coll_batch[k] = torch.concat([torch.tensor(frame.atoms[k]) for frame in batch])

    seg_m = torch.cumsum(coll_batch[alias.natoms], dim=0)
    seg_m = torch.cat(
        [torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0
    )  # prepend 0 to seg_m
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[alias.natoms], dim=0
    )
    coll_batch[alias.idx_m] = idx_m

    return coll_batch


@functional_datapipe("collate_frames")
class CollateFrames(Collator):
    def __init__(self, datapipe, **kwargs):
        super().__init__(datapipe, collate_fn=_collate, **kwargs)


@functional_datapipe("calc_nblist")
class CalcNBList(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, cutoff: float):
        self.dp = source_dp
        self.cutoff = cutoff

    def __iter__(self):
        nblist = TorchNeighborList(cutoff=self.cutoff)
        for d in self.dp:
            d = nblist(d)
            yield d
