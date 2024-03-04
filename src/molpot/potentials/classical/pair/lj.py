# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-26
# version: 0.0.1

from .base import PairBase
from torch import nn
from molpot import Alias

import torch

class LJ126(PairBase):

    def __init__(self, ntypes:int, cutoff:int):
        super().__init__("LJ126", ntypes)

        self.init_params('eps', (ntypes, ntypes))
        self.init_params('sig', (ntypes, ntypes))

    def forward(self, tensor):
        print(f"idx_i req grad: {tensor[Alias.idx_i].requires_grad}")
        print(f"idx_j req grad: {tensor[Alias.idx_j].requires_grad}")
        idx_i = tensor[Alias.idx_i]
        idx_j = tensor[Alias.idx_j]
        types = tensor[Alias.atype]
        eps = self.eps[types[idx_i], types[idx_j]]
        sig = self.sig[types[idx_i], types[idx_j]]
        offsets = tensor[Alias.offsets]
        R = tensor[Alias.R]
        Rij = R[idx_i] - R[idx_j] + offsets
        dij = torch.norm(Rij, dim=1)
        pairwise_energy = 4 * eps * ((sig / dij)**12 - (sig / dij)**6)
        per_atom_energy = torch.zeros((R.shape[0], 1)).index_add(0, idx_i, pairwise_energy.reshape(-1, 1))
        tensor[Alias.per_atom_energy] += per_atom_energy / 2
        return tensor