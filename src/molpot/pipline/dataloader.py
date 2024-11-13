from typing import Sequence

import torch
from torch.utils.data import DataLoader

from molpot import Config, Frame, alias

def cancel_batch(tensor_or_nested_tensor: torch.Tensor):
    if isinstance(tensor_or_nested_tensor, torch.Tensor):
        if tensor_or_nested_tensor.is_nested:
            return torch.concat(tensor_or_nested_tensor)
        else:
            return tensor_or_nested_tensor.reshape(-1, *tensor_or_nested_tensor.shape[2:])
    return tensor_or_nested_tensor.apply(cancel_batch, batch_size=[], call_on_nested=True)

def _collate_frame(batch: Sequence[Frame]):

    coll_batch = Frame.maybe_dense_stack(batch).densify()
    batch_size = int(coll_batch.batch_size.numel())

    if alias.n_atoms not in coll_batch:
        n_atoms = torch.tensor(
            [len(frame[alias.R]) for frame in batch], dtype=Config.itype
        )
    else:
        n_atoms = coll_batch[alias.n_atoms]

    atom_batch_mask = (
        torch.cat([torch.full((t,), fill_value=i) for i, t in enumerate(n_atoms)])
        .reshape(batch_size, -1)
        .to(torch.int64)
    )  # scatter

    atomistic_offset = torch.cumsum(torch.flatten(n_atoms), dim=0).to(Config.itype)[:-1]
    atomistic_offset = torch.nested.as_nested_tensor([torch.tensor([0], dtype=atomistic_offset.dtype), *torch.split(atomistic_offset, 1)])

    coll_frame = coll_batch.apply(cancel_batch, batch_size=[])
    for key in [alias.pair_i, alias.pair_j, alias.bond_i, alias.bond_j]:
        if key in coll_batch:
            coll_frame[key] = torch.where()
    coll_frame[alias.atom_batch_mask] = atom_batch_mask.flatten()

    return coll_frame


class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = _collate_frame
        super().__init__(*args, **kwargs)
