from torch.utils.data import DataLoader
from typing import Sequence
from molpot import Frame, alias
import torch


def _collate_frame(batch: Sequence[Frame]):

    coll_batch = Frame.maybe_dense_stack(batch).densify()

    # calculate atomic batch mask:
    # slice atoms from different frame
    coll_batch[alias.atom_batch_mask] = torch.nested.nested_tensor(
        [
            torch.full((t,), fill_value=i)
            for i, t in enumerate(coll_batch[alias.n_atoms])
        ]
    )

    # offset of atom
    atomistic_offset = torch.cumsum(coll_batch[alias.n_atoms], dim=0)
    atomistic_offset = torch.cat(
        [torch.zeros((1,), dtype=atomistic_offset.dtype), atomistic_offset], dim=0
    )  # prepend 0 to atomistic_offset
    for key in [alias.pair_i, alias.pair_j, alias.bond_i, alias.bond_j]:
        if key in coll_batch:
            coll_batch[key] = torch.torch.nested.nested_tensor(
                [d[key] + off for d, off in zip(batch, atomistic_offset)]
            )

    new_frame = Frame()

    for category, td in coll_batch.items():
        for key, value in td.items():
            flatten_tensor = torch.cat(value.unbind())
            new_frame[category, key] = flatten_tensor

    return coll_batch


class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=_collate_frame, *args, **kwargs)
