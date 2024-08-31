from functools import partial

import torch
from molpot_op import get_neighbor_pairs
from torch import nn

from molpot import alias


class NeighborList(nn.Module):

    def __init__(self, cutoff:float, max_num_pairs:int=-1, check_errors:bool=False):
        super().__init__()
        self.cutoff = cutoff
        self.kernel = partial(get_neighbor_pairs, max_num_pairs=max_num_pairs, check_errors=check_errors)

    def forward(self, inputs):

        xyz = inputs[alias.xyz]
        if alias.cell not in inputs:
            box = None
        else:
            box = inputs[alias.cell]

        pairs, deltas, distances, n_pairs = self.kernel(positions=xyz, box_vectors=box, cutoff=self.cutoff)

        inputs[alias.pair_i] = pairs[0]
        inputs[alias.pair_j] = pairs[1]
        inputs[alias.pair_diff] = deltas
        inputs[alias.pair_dist] = distances

        inputs[alias.n_pairs] = n_pairs
        return inputs
