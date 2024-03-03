import torch
import torch.nn as nn

import molpot as mp

from .nnp.ops import index_add


class NNPotential(nn.Sequential):

    def __init__(self, name, *potentials):
        super().__init__(*potentials)
        self._name = name

    def cite(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    def forward(self, inputs):

        inputs = super().forward(inputs)
        idx_m = inputs["_idx_m"]
        maxm = torch.max(idx_m) + 1
        per_atom_energy = inputs["per_atom_energy"]
        inputs[mp.Alias.energy] = index_add(
            torch.squeeze(per_atom_energy, -1), 0, idx_m, dim_size=maxm
        )

        de_drij = torch.autograd.grad(
                per_atom_energy,
                inputs[mp.Alias.Rij],
                grad_outputs=torch.ones_like(per_atom_energy),
                retain_graph=True,
                create_graph=True,
            )[0]
        assert de_drij is not None
        # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff
        forces = (
            torch.zeros_like(inputs[mp.Alias.R])
            .index_add(0, inputs[mp.Alias.idx_i], de_drij)
            .index_add(0, inputs[mp.Alias.idx_j], -de_drij)
        )
        inputs[mp.Alias.forces] = forces
        return inputs

    def grad(
        self,
    ):
        return torch.autograd.grad(
            self._modules,
        )[0]


class Potential(nn.Module):

    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._name = name
