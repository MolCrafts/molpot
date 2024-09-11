from functools import partial

import torch
from molpot_op import get_neighbor_pairs
from torch import nn

from molpot import alias


class NeighborList(nn.Module):

    def __init__(
        self,
        cutoff: float,
        max_num_pairs: int = -1,
        check_errors: bool = False,
        index_dtype: torch.dtype = torch.int32,
        exclude_ii: bool = True,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.kernel = partial(
            get_neighbor_pairs, max_num_pairs=max_num_pairs, check_errors=check_errors
        )
        self.index_dtype = index_dtype
        self.exclude_ii = exclude_ii

    def forward(self, inputs):

        xyz = inputs[alias.xyz]
        if alias.cell not in inputs:
            box = None
        else:
            box = inputs[alias.cell]

        pairs, deltas, distances, n_pairs = self.kernel(
            positions=xyz, box_vectors=box, cutoff=self.cutoff
        )
        pairs = pairs.to(dtype=self.index_dtype)
        if self.exclude_ii:
            mask = ~torch.isnan(distances)
            pairs = pairs[:, mask]
            deltas = deltas[mask]
            distances = distances[mask]
            n_pairs = mask.sum(dim=0)

        inputs[alias.pair_i] = pairs[0]
        inputs[alias.pair_j] = pairs[1]
        inputs[alias.pair_diff] = deltas
        inputs[alias.pair_dist] = distances

        inputs[alias.n_pairs] = torch.tensor([n_pairs], dtype=torch.int32)
        return inputs
