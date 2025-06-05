from functools import partial

import torch
from molpot_op import get_neighbor_pairs
from torch import nn

from molpot import alias

from .base import Process, ProcessType

class NeighborList(Process):

    type = ProcessType.ONE

    def __init__(
        self,
        cutoff: float,
        max_num_pairs: int = -1,
        check_errors: bool = False,
        index_dtype: torch.dtype = torch.int32
    ):
        super().__init__()
        self.cutoff = cutoff
        self.kernel = partial(
            get_neighbor_pairs, max_num_pairs=max_num_pairs, check_errors=check_errors
        )
        self.index_dtype = index_dtype

    def forward(self, inputs):
        """ calculate neighbor list. The pair distance is calculated by xyz[pair_j] - xyz[pair_i]

        Args:
            inputs (tensordict): required keys: alias.R, alias.cell

        Returns:
            tensordict: alias.pair_i, alias.pair_j, alias.pair_diff, alias.pair_dist, alias.n_pairs
        """
        xyz = inputs[alias.R]
        if alias.cell not in inputs:
            box = None
        else:
            box = inputs[alias.cell]
        pairs, deltas, distances, n_pairs = self.kernel(
            positions=xyz, box_vectors=box, cutoff=self.cutoff
        )
        pairs = pairs.to(dtype=self.index_dtype)

        mask = ~torch.isnan(distances)
        pairs = pairs[:, mask]
        deltas = deltas[mask]
        distances = distances[mask]
        n_pairs = mask.sum(dim=0)

        inputs[alias.pair_i] = pairs[1]
        inputs[alias.pair_j] = pairs[0]
        inputs[alias.pair_diff] = deltas.requires_grad_(True)
        inputs[alias.pair_dist] = distances.requires_grad_(True)

        inputs[alias.n_pairs] = torch.tensor([n_pairs], dtype=torch.int32)
        return inputs
