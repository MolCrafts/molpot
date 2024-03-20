# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from typing import Sequence

import molpy as mp
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import Collator

import molpot as mpot
from molpot import Alias, Config


def _collate_dict(batch: Sequence[dict]):
    props_keys = batch[0].keys()
    coll_batch = {}
    for k in props_keys:
        coll_batch[k] = torch.cat(
            [torch.atleast_1d(frame[k]) for frame in batch], 0
        ).to(Config.device)

    idx_m = torch.repeat_interleave(
        torch.arange(len(batch), device=Config.device),
        repeats=coll_batch[mpot.Alias.n_atoms],
        dim=0,
    )
    coll_batch[mpot.Alias.idx_m] = idx_m
    
    to_be_offset_keys = [Alias.idx_i, Alias.idx_j]
    offset_keys = []
    for key in to_be_offset_keys:
        if key in coll_batch:
            offset_keys.append(key)

    if offset_keys:
        seg_m = torch.cumsum(coll_batch[mpot.Alias.n_atoms], dim=0)
        seg_m = torch.cat(
            [torch.zeros((1,), dtype=seg_m.dtype, device=Config.device), seg_m], dim=0
        )  # prepend 0 to seg_m
        for key in offset_keys:
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, seg_m)], dim=0
            ).to(Config.device)

    return coll_batch


@functional_datapipe("collate_data")
class CollateData(Collator):
    def __init__(self, datapipe, **kwargs):
        super().__init__(datapipe, collate_fn=_collate_dict, **kwargs)
