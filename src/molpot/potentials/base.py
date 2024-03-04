import torch
import torch.nn as nn

import molpot as mp

class Potentials(nn.Sequential):

    def __init__(self, name, *potentials, derive_energy: bool = True):
        super().__init__(*potentials)
        self._name = name
        self.derive_energy = derive_energy

    def cite(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name
    
    def forward(self, inputs:dict[str, dict]):

        if mp.Alias.energy not in inputs:
            inputs[mp.Alias.energy] = 0

        inputs = super().forward(inputs)
        if self.derive_energy:
            # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff 
            de_drij = torch.autograd.grad(torch.sum(inputs[mp.Alias.energy]), inputs[mp.Alias.R], retain_graph=True)[0]
            inputs[mp.Alias.forces] = de_drij

        return inputs

class Potential(nn.Module):

    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._name = name
