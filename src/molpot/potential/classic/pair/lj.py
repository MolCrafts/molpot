from typing import Callable
import torch
from torch import nn

from molpot import alias

from torch.nn.init import xavier_uniform_

class LJ126(nn.Module):

    in_keys = [alias.pair_dist]

    @staticmethod
    def E(sig, eps, d_ij):
        power_6 = torch.pow(sig / d_ij, 6)
        power_12 = torch.square(power_6)
        return 4 * eps * (power_12 - power_6)

    @staticmethod
    def F(sig, eps, r_ij):
        """Force calculation for Lennard-Jones 12-6 potential"""
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
        power_6 = torch.pow(sig / d_ij, 6)
        power_12 = torch.square(power_6)
        return 24 * eps * (2 * power_12 - power_6) * r_ij / torch.square(d_ij)

    def __init__(self, sig: torch.Tensor|None = None, eps: torch.Tensor|None = None):
        super().__init__()
        self.eps = eps
        self.sig = sig
        self.register_forward_pre_hook(self._inter_parameters, with_kwargs=False)

    def _inter_parameters(self, module, args, kwargs):
        pair_dist, atom_types = args
        if self.sig is None:
            self.sig = torch.zeros((atom_types, atom_types))
        if self.eps is None:
            self.eps = torch.ones((atom_types, atom_types))
        


    def forward(self, pair_dist, atom_types):

        self.energy(inputs, outputs)

        return inputs, outputs

    def energy(self, pair_dist, atom_types):

        power_6 = torch.pow(sig / d_ij, 6)
        power_12 = torch.square(power_6)
        return 4 * eps * (power_12 - power_6)

    def forces(self, inputs, outputs):

        r_ij = outputs["pairs"][alias.pair_diff]
        outputs["pairs"]["lj126_forces"] = self.F(self.sig, self.eps, r_ij)

        return inputs, outputs
