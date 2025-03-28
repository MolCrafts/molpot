from functools import reduce
from typing import Iterator, Sequence

import torch
import torchdata.nodes as tn
from torch.utils.data import (Dataset, RandomSampler, SequentialSampler,
                              default_collate)

from molpot import Config, Frame, alias

config = Config()

def compose(*funcs):
    """Compose a list of functions into a single function."""
    return lambda x: reduce(lambda v, f: f(v), funcs, x)


def cancel_batch(tensor_or_nested_tensor: torch.Tensor):
    if isinstance(tensor_or_nested_tensor, torch.Tensor):
        if tensor_or_nested_tensor.is_nested:
            return torch.concat(tensor_or_nested_tensor.unbind())
        else:
            return tensor_or_nested_tensor.reshape(
                -1, *tensor_or_nested_tensor.shape[2:]
            )
    return tensor_or_nested_tensor.apply(
        cancel_batch, batch_size=[], call_on_nested=True
    )


def _compact_collate(batch: Sequence[Frame]):
    """collate a batch of frames into a single frame, with batch masks and offsets. no nested tensors.

    Args:
        batch (Sequence[Frame]): _description_

    Returns:
        _type_: _description_
    """
    coll_batch = Frame.maybe_dense_stack(batch).densify()
    # batch_size = int(coll_batch.batch_size.numel())

    if alias.n_atoms not in coll_batch:
        n_atoms = torch.tensor(
            [len(frame[alias.R]) for frame in batch], dtype=config.itype
        )
    else:
        n_atoms = coll_batch[alias.n_atoms]

    atom_batch = torch.cat(
        [torch.full((t,), fill_value=i) for i, t in enumerate(n_atoms)]
    ).to(config.itype)

    atom_offset = torch.zeros(len(n_atoms), dtype=config.itype)
    torch.cumsum(torch.flatten(n_atoms)[:-1], dim=0, out=atom_offset[1:]).to(
        config.itype
    )

    coll_frame = coll_batch.apply(cancel_batch, batch_size=[])
    coll_frame[alias.atom_batch] = atom_batch
    coll_frame[alias.atom_offset] = atom_offset

    if alias.pairs in coll_batch:
        n_pairs = torch.tensor(
            [len(frame[alias.pair_i]) for frame in batch], dtype=config.itype
        )
        pair_batch = torch.cat(
            [
                torch.full((t,), fill_value=i)
                for i, t in enumerate([len(frame[alias.pair_i]) for frame in batch])
            ]
        )
        pair_offset = torch.zeros(len(n_pairs), dtype=config.itype)
        torch.cumsum(torch.flatten(n_pairs)[:-1], dim=0, out=pair_offset[1:]).to(
            config.itype
        )
        coll_frame[alias.pair_i] = (
            torch.concat(coll_batch[alias.pair_i].unbind()) + atom_offset[pair_batch]
        )
        coll_frame[alias.pair_j] = (
            torch.concat(coll_batch[alias.pair_j].unbind()) + atom_offset[pair_batch]
        )
        coll_frame[alias.pair_batch] = pair_batch
        coll_frame[alias.pair_offset] = pair_offset

    return coll_frame


def _nested_collate(batch: Sequence[Frame]):
    return Frame.from_frames(batch).densify()


class MapAndCollate:
    """A simple transform that takes a batch of indices, maps with dataset, and then applies
    collate.
    TODO: make this a standard utility in torchdata.nodes
    """

    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __call__(self, batch_of_indices: list[int]):
        batch = [self.dataset[i] for i in batch_of_indices]
        return self.collate_fn(batch)


class DataLoader(tn.Loader):

    def __init__(
        self,
        dataset: Dataset | Iterator[Frame],
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn=_compact_collate,
    ):

        # Start with a sampler, since caller did not provide one
        sampler = RandomSampler(dataset, generator=config.get_generator()) if shuffle else SequentialSampler(dataset)
        # Sampler wrapper converts a Sampler to a BaseNode
        node = tn.SamplerWrapper(sampler)

        # Now let's batch sampler indices together
        node = tn.Batcher(node, batch_size=batch_size, drop_last=drop_last)

        # Create a Map Function that accepts a list of indices, applies getitem to it, and
        # then collates them
        map_and_collate = MapAndCollate(dataset, collate_fn or default_collate)

        # MapAndCollate is doing most of the heavy lifting, so let's parallelize it. We could
        # choose process or thread workers. Note that if you're not using Free-Threaded
        # Python (eg 3.13t) with -Xgil=0, then multi-threading might result in GIL contention,
        # and slow down training.
        if num_workers > 1:
            node = tn.ParallelMapper(
                node,
                map_fn=map_and_collate,
                num_workers=num_workers,
                method="process",  # Set this to "thread" for multi-threading
                in_order=True,
            )
            prefetch_factor = num_workers * 2
        else:
            node = tn.Mapper(node, map_fn=map_and_collate)
            prefetch_factor = 1

        # Optionally apply pin-memory, and we usually do some pre-fetching
        if pin_memory:
            node = tn.PinMemory(node)
        node = tn.Prefetcher(node, prefetch_factor=prefetch_factor)

        # Note that node is an iterator, and once it's exhausted, you'll need to call .reset()
        # on it to start a new Epoch.
        # Insteaad, we wrap the node in a Loader, which is an iterable and handles reset. It
        # also provides state_dict and load_state_dict methods.

        super().__init__(node)
