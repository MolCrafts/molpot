from torch.utils.data import DataLoader, default_collate
from typing import Sequence
from molpot import Frame, alias
import torch
from tensordict import TensorDict


def _collate_frame(batch: Sequence[Frame]):

    coll_batch = Frame.maybe_dense_stack(batch)
    batch_size = int(coll_batch.batch_size.numel())

    if alias.n_atoms not in coll_batch:
        n_atoms = torch.tensor([len(frame[alias.R]) for frame in batch], dtype=int)
    else:
        n_atoms = coll_batch[alias.n_atoms]

    coll_batch[alias.atom_batch_mask] = torch.cat(
        [torch.full((t,), fill_value=i) for i, t in enumerate(n_atoms)]
    ).reshape(batch_size, -1)

    atomistic_offset = torch.cumsum(n_atoms.squeeze(), dim=0)
    atomistic_offset = torch.cat(
        (torch.zeros((1,), dtype=atomistic_offset.dtype), atomistic_offset), dim=0
    )
    for key in [alias.pair_i, alias.pair_j, alias.bond_i, alias.bond_j]:
        if key in coll_batch:
            coll_batch[key] = coll_batch[key] + atomistic_offset[coll_batch[alias.atom_batch_mask]]
            # torch.nested.nested_tensor(
            #     [d[key] + off for d, off in zip(batch, atomistic_offset)]
            # )

    return coll_batch


class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=_collate_frame, *args, **kwargs)
