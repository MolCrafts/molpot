from typing import Sequence
from torchdata.datapipes.iter import IterDataPipe, Collator
from torchdata.datapipes import functional_datapipe
import molpy as mp
import molpot as mpot
import torch
from molpot import kw

__all__ = [
    "XYZReader",
]


@functional_datapipe("read_xyz")
class XYZReader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, keywords: mp.Keywords):
        super().__init__()
        self.source_dp = source_dp
        self.keywords = keywords

    def __iter__(self):
        local_kw = self.keywords
        for d in self.source_dp:
            frame = mp.Frame()

            lines = d[1].readlines()
            frame[kw.natoms] = int(lines[0])
            props_line = lines[1].split()[1:]
            frame["index"] = int(props_line[0])
            for prop, p in zip(local_kw.get_aliases(), props_line[1:]):
                if prop in local_kw:
                    src_unit = local_kw.get_unit(prop)
                    dst_unit = kw.get_unit(prop)
                    frame[prop] = mp.units.convert(float(p), src_unit, dst_unit)
                else:
                    frame[prop] = float(p)

            frame.atoms[kw.xyz] = [
                [i.replace("*^", "E") for i in line.split()[1:4]]
                for line in lines[2:-3]
            ]
            frame.atoms[kw.Z] = [
                mp.Element.get_atomic_number_by_symbol(line.split()[0])
                for line in lines[2:-3]
            ]

            yield frame

@functional_datapipe("collate_frames")
class CollateFrames(Collator):
    def __init__(self, datapipe, **kwargs):
        super().__init__(datapipe, collate_fn=self.collate, **kwargs)

    def _collate(self, batch: Sequence[mp.Frame]):
        coll_batch = {}

        props_keys = batch[0]._props.keys()
        atoms_keys = batch[0].atoms.keys()

        for k in props_keys:
            coll_batch[k] = [frame[k] for frame in batch]

        for k in atoms_keys:
            coll_batch[k] = [frame.atoms[k] for frame in batch]

        seg_m = torch.cumsum(coll_batch[kw.natoms], dim=0)
        seg_m = torch.cat(
            [torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0
        )  # prepend 0 to seg_m
        idx_m = torch.repeat_interleave(
            torch.arange(len(batch)), repeats=coll_batch[kw.natoms], dim=0
        )
        coll_batch[kw.idx_m] = idx_m

# @functional_datapipe("calc_nblist")
# def CalcNBList(IterDataPipe):
#     def __init__(self, source_dp: IterDataPipe, cutoff:float, **kwargs):
#         self.dp = source_dp
#         self.cutoff = cutoff

#     def __iter__(self):
#         nblist = mpot.transform.TorchNeighborList(cutoff=self.cutoff)
#         for d in self.source_dp:
#             xyz = d[kw.R]
#             nblist(xyz)
#             yield frame
