import torch
import torch.nn as nn

import molpot as mpot
import molpy as mp


class Potential(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name
    
class PotentialDict(nn.ModuleDict, Potential):

    pass
    
class PotentialSeq(Potential):

    def __init__(self, name, *modules):
        super().__init__(name)
        self.seq = nn.Sequential(*modules)

    def forward(self, inputs, outputs):
        for module in self.seq:
            inputs, outputs = module(inputs, outputs)
        return inputs, outputs
    
class ForceField(mp.ForceField):

    def get_potential(self):

        potentials = []

        for style in self.pairstyles:
            
            params = style.get_params()
            potential = mpot.potential.get_classic_potental(style.name, 'pair')
            potentials.append(potential(**params))

        for style in self.bondstyles:
            params = style.get_params()
            potential = mpot.potential.get_classic_potental(style.name, 'bond')
            potentials.append(potential(**params))

        return potentials