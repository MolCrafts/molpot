import torch
import torch.nn as nn
import molpot as mp

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