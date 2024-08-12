# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from typing import Sequence

import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import Collator

import molpot as mpot
from molpot import Config
from molpot.alias import *


def _collate_frame(batch: Sequence[dict]):
    probe_frame = batch[0]
    coll_batch = mpot.Frame()
    for k in list(probe_frame.keys()):
        for kk in list(probe_frame[k].keys()):
            coll_batch[k, kk] = torch.cat(
                [torch.atleast_1d(frame[k, kk]) for frame in batch], 0
            )

    batch_mask = torch.repeat_interleave(
        torch.arange(len(batch)),
        repeats=coll_batch[n_atoms],
        dim=0,
    )
    coll_batch[atom_batch_mask] = batch_mask

    atomistic_offset = torch.cumsum(coll_batch[n_atoms], dim=0)
    atomistic_offset = torch.cat(
        [torch.zeros((1,), dtype=atomistic_offset.dtype), atomistic_offset], dim=0
    )  # prepend 0 to atomistic_offset
    for key in [pair_i, pair_j]:
        if key in coll_batch:
            coll_batch[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, atomistic_offset)], dim=0
            )
    coll_batch[n_batches] = len(batch)
    return coll_batch


@functional_datapipe("collate_frame")
class CollateFrame(Collator):
    def __init__(self, datapipe, **kwargs):
        super().__init__(datapipe, collate_fn=_collate_frame, **kwargs)
