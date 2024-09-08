from torch.utils.data import DataLoader
from typing import Sequence
from molpot import Frame, alias
import torch


def _collate_frame(batch: Sequence[Frame]):

    coll_batch = Frame.maybe_dense_stack(batch)
    coll_frame = Frame()

    # calculate atomic batch mask:
    # slice atoms from different frame

    if alias.n_atoms not in coll_batch:
        n_atoms = len(coll_batch[0][alias.R])
        n_atoms_list = torch.full((len(coll_batch),), n_atoms, dtype=torch.int64)
        assert len(coll_batch[0][alias.R]) == len(
            coll_batch[1][alias.R]
        ), "Number of atoms should be the same in each frame if n_atoms is provided for each frame"
    else:
        n_atoms_list = coll_batch[alias.n_atoms]

    coll_frame[alias.atom_batch_mask] = torch.cat(
        [torch.full((t,), fill_value=i) for i, t in enumerate(n_atoms_list)]
    )

    # offset of atom
    atomistic_offset = torch.cumsum(n_atoms_list.squeeze(), dim=0)
    atomistic_offset = torch.cat(
        (torch.zeros((1,), dtype=atomistic_offset.dtype), atomistic_offset), dim=0
    )  # prepend 0 to atomistic_offset
    for key in [alias.pair_i, alias.pair_j, alias.bond_i, alias.bond_j]:
        if key in coll_batch:
            coll_frame[key] = torch.cat(
                [d[key] + off for d, off in zip(batch, atomistic_offset)]
            )


    for category, td in coll_batch.items():
        for key, value in td.items():
            # merge batch, batch dim is the first dim
            flatten_tensor = torch.flatten(value, start_dim=0, end_dim=1)
            coll_frame[category, key] = flatten_tensor

    return coll_frame


class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=_collate_frame, *args, **kwargs)
