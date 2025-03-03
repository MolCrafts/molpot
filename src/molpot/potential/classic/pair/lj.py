from typing import Callable
import torch
from torch.nn import Module

from molpot import alias

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

class LJ126(LazyModuleMixin, Module):

    in_keys = [alias.pair_diff, alias.atom_types]
    out_keys = [("predicts", "lj126_energy"), ("predicts", "lj126_forces")]

    def __init__(self, sig: torch.Tensor|None = None, eps: torch.Tensor|None = None, device=None, dtype=None):
        super().__init__()
        self.eps = eps if isinstance(eps, torch.Tensor) else UninitializedParameter(device=device, dtype=dtype)
        self.sig = sig if isinstance(eps, torch.Tensor) else UninitializedParameter(device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        self.sig = torch.nn.init.uniform_(self.sig, a=0.1, b=1.0)
        self.eps = torch.nn.init.uniform_(self.eps, a=0.1, b=1.0)

    def initialize_parameters(self, pair_diff, atom_types) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                n_types = atom_types.max().item() + 1
                self.sig.materialize((n_types, n_types))
                self.eps.materialize((n_types, n_types))
                self.reset_parameters()
        
    def forward(self, pair_diff, atom_types):

        return self.energy(pair_diff, atom_types), self.forces(pair_diff, atom_types)

    def energy(self, pair_diff, atom_types):

        sig = self.sig[atom_types[:, 0], atom_types[:, 1]]
        eps = self.eps[atom_types[:, 0], atom_types[:, 1]]

        power_6 = torch.pow(sig / pair_diff, 6)
        power_12 = torch.square(power_6)
        return 4 * eps * (power_12 - power_6)

    def forces(self, pair_diff, atom_types):

        sig = self.sig[atom_types[:, 0], atom_types[:, 1]]
        eps = self.eps[atom_types[:, 0], atom_types[:, 1]]

        d_ij = torch.norm(pair_diff, dim=-1, keepdim=True)
        power_6 = torch.pow(sig / d_ij, 6)
        power_12 = torch.square(power_6)
        return 24 * eps * (2 * power_12 - power_6) * pair_diff / torch.square(d_ij)
