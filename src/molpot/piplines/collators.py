# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from typing import Sequence
from torchdata.datapipes.iter import Collator
import molpy as mp
import torch
import molpot as mpot
from torchdata.datapipes import functional_datapipe

__all__ = ["CollateFrames"]

def _collate(batch: Sequence[mp.Frame]):

    coll_batch = {}

    props_keys = batch[0]._props.keys()
    atoms_keys = batch[0].atoms.keys()

    for k in props_keys:
        coll_batch[k] = torch.concat(
            [torch.atleast_1d(torch.tensor(frame[k])) for frame in batch]
        )

    for k in atoms_keys:
        coll_batch[k] = torch.concat([torch.tensor(frame.atoms[k]) for frame in batch])

    seg_m = torch.cumsum(coll_batch[mpot.alias.natoms], dim=0)
    seg_m = torch.cat(
        [torch.zeros((1,), dtype=seg_m.dtype), seg_m], dim=0
    )  # prepend 0 to seg_m
    idx_m = torch.repeat_interleave(
        torch.arange(len(batch)), repeats=coll_batch[mpot.alias.natoms], dim=0
    )
    coll_batch[mpot.alias.idx_m] = idx_m

    return coll_batch


@functional_datapipe("collate_frames")
class CollateFrames(Collator):
    def __init__(self, datapipe, **kwargs):
        super().__init__(datapipe, collate_fn=_collate, **kwargs)

