from typing import Callable
import torch
from torch.nn import Module

from molpot import alias

from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


class LJ126(LazyModuleMixin, Module):

    in_keys = [alias.pair_i, alias.pair_j, alias.pair_diff, alias.atom_type]
    out_keys = [
        ("predicts", "lj126_energy"),
    ]

    def __init__(
        self,
        sig: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
        calc_forces: bool = False,
    ):
        super().__init__()
        self.eps = eps if isinstance(eps, torch.Tensor) else UninitializedParameter()
        self.sig = sig if isinstance(eps, torch.Tensor) else UninitializedParameter()
        if calc_forces:
            self.out_keys.append(("predicts", "lj126_forces"))
        self.calc_forces = calc_forces

    def reset_parameters(self) -> None:
        self.sig = torch.nn.init.uniform_(self.sig, a=0.1, b=1.0)
        self.eps = torch.nn.init.uniform_(self.eps, a=0.1, b=1.0)

    def initialize_parameters(self, pair_i, pair_j, pair_diff, atom_types) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                n_types = atom_types.max().item() + 1
                self.sig.materialize((n_types, n_types))
                self.eps.materialize((n_types, n_types))
                self.reset_parameters()

    def forward(self, pair_i, pair_j, pair_diff, atom_types):
        pair_dist = torch.norm(pair_diff, dim=-1)

        energy = self.energy(pair_i, pair_j, pair_dist, atom_types)

        if self.calc_forces:
            return energy, self.forces(pair_i, pair_j, pair_diff, pair_dist, atom_types)
        return energy

    def energy(self, pair_i, pair_j, pair_dist, atom_types):
        sig = self.sig[atom_types[pair_i], atom_types[pair_j]]
        eps = self.eps[atom_types[pair_i], atom_types[pair_j]]

        power_6 = torch.pow(sig / pair_dist, 6)
        power_12 = torch.square(power_6)
        return 4 * eps * (power_12 - power_6)

    def forces(self, pair_i, pair_j, pair_diff, pair_dist, atom_types):

        sig = self.sig[atom_types[pair_i], atom_types[pair_j]]
        eps = self.eps[atom_types[pair_i], atom_types[pair_j]]

        power_6 = torch.pow(sig / pair_dist, 6)
        power_12 = torch.square(power_6)
        unit_force = 24 * eps * (2 * power_12 - power_6) / torch.square(pair_dist)
        return unit_force.unsqueeze(-1) * pair_diff


class Stockmayer(LJ126):

    def __init__(
        self,
        sig: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        xi: torch.Tensor | None = None,
        calc_forces: bool = False,
    ):
        super().__init__(sig, eps, calc_forces)
        self.mu = mu if isinstance(mu, torch.Tensor) else UninitializedParameter()
        self.xi = xi if isinstance(xi, torch.Tensor) else UninitializedParameter()
        self.out_keys.append(("predicts", "stockmayer_energy"))

    def reset_parameters(self) -> None:
        self.sig = torch.nn.init.uniform_(self.sig, a=0.1, b=1.0)
        self.eps = torch.nn.init.uniform_(self.eps, a=0.1, b=1.0)
        self.mu = torch.nn.init.uniform_(self.mu, a=0.1, b=1.0)
        self.xi = torch.nn.init.uniform_(self.xi, a=0.1, b=1.0)

    def initialize_parameters(self, pair_i, pair_j, pair_diff, atom_types) -> None:  # type: ignore[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                n_types = atom_types.max().item() + 1
                self.sig.materialize((n_types, n_types))
                self.eps.materialize((n_types, n_types))
                self.mu.materialize((n_types, ))
                self.xi.materialize((1))
                self.reset_parameters()

    def energy(self, pair_i, pair_j, pair_dist, atom_types):
        lj_energy = super().energy(pair_i, pair_j, pair_dist, atom_types)
        mu1 = self.mu[atom_types[pair_i]]
        mu2 = self.mu[atom_types[pair_j]]
        xi = self.xi

        return lj_energy + mu1 * mu2 * xi / torch.square(pair_dist)