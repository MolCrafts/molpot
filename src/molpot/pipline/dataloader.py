from typing import Sequence

import torch
from torch.utils.data import DataLoader

from molpot import Config, Frame, alias

def cancel_batch(tensor_or_nested_tensor: torch.Tensor):
    if isinstance(tensor_or_nested_tensor, torch.Tensor):
        if tensor_or_nested_tensor.is_nested:
            return torch.concat([td for td in tensor_or_nested_tensor])
        else:
            return tensor_or_nested_tensor.reshape(-1, *tensor_or_nested_tensor.shape[2:])
    return tensor_or_nested_tensor.apply(cancel_batch, batch_size=[], call_on_nested=True)

def _collate_frame(batch: Sequence[Frame]):

    coll_batch = Frame.maybe_dense_stack(batch).densify()
    # batch_size = int(coll_batch.batch_size.numel())

    if alias.n_atoms not in coll_batch:
        n_atoms = torch.tensor(
            [len(frame[alias.R]) for frame in batch], dtype=Config.itype
        )
    else:
        n_atoms = coll_batch[alias.n_atoms]

    atom_batch_mask = torch.cat([torch.full((t,), fill_value=i) for i, t in enumerate(n_atoms)]).to(alias.atoms_ns['atom_batch_mask'].dtype)

    atom_offset = torch.zeros(len(n_atoms), dtype=Config.itype)
    torch.cumsum(torch.flatten(n_atoms)[:-1], dim=0, out=atom_offset[1:]).to(Config.itype)

    if alias.pairs in coll_batch:
        n_pairs = torch.tensor(
            [len(frame[alias.pair_i]) for frame in batch], dtype=Config.itype
        )
        pair_batch_mask = torch.cat([torch.full((t,), fill_value=i) for i, t in enumerate([len(frame[alias.pair_i]) for frame in batch])])
        pair_offset = torch.zeros(len(n_pairs), dtype=Config.itype)
        torch.cumsum(torch.flatten(n_pairs)[:-1], dim=0, out=pair_offset[1:]).to(Config.itype)


    coll_frame = coll_batch.apply(cancel_batch, batch_size=[])
    coll_frame[alias.pair_i] = coll_frame[alias.pair_i] + atom_offset[pair_batch_mask]
    coll_frame[alias.pair_j] = coll_frame[alias.pair_j] + atom_offset[pair_batch_mask]
    coll_frame[alias.atom_batch_mask] = atom_batch_mask
    coll_frame[alias.atom_offset] = atom_offset
    coll_frame[alias.pair_batch_mask] = pair_batch_mask
    coll_frame[alias.pair_offset] = pair_offset
    return coll_frame


class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = _collate_frame
        super().__init__(*args, **kwargs)
