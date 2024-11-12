from torch.utils.data import DataLoader, default_collate
from typing import Sequence
from molpot import Frame, alias, Config
import torch
from tensordict import TensorDict
from typing import TypeVar, Optional, Union, List, Tuple, Dict, Callable, Iterable
from .dataset import Dataset
from torch.utils.data import Sampler, IterableDataset

_T_co = TypeVar("_T_co", covariant=True)


def _collate_frame(batch: Sequence[Frame]):

    coll_batch = Frame.maybe_dense_stack(batch).densify()
    batch_size = int(coll_batch.batch_size.numel())

    if alias.n_atoms not in coll_batch:
        n_atoms = torch.tensor([len(frame[alias.R]) for frame in batch], dtype=Config.itype)
    else:
        n_atoms = coll_batch[alias.n_atoms]

    atom_batch_mask = torch.cat(
        [torch.full((t,), fill_value=i) for i, t in enumerate(n_atoms)]
    ).reshape(batch_size, -1).to(torch.int64)  # scatter

    atomistic_offset = torch.cumsum(
        torch.cat([torch.tensor([0]), torch.flatten(n_atoms)]), dim=0
    ).to(Config.itype)

    for key in [alias.pair_i, alias.pair_j, alias.bond_i, alias.bond_j]:
        if key in coll_batch:
            coll_batch[key] = coll_batch[key] + atomistic_offset[:-1][:, None]

    coll_frame = coll_batch.apply(lambda x: x.reshape(-1, *x.shape[2:]), batch_size=[])
    coll_frame[alias.atom_batch_mask] = atom_batch_mask.flatten()

    return coll_frame


class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = _collate_frame
        super().__init__(*args, **kwargs)