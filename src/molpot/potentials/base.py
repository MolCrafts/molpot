import torch
import torch.nn as nn

class NNPotential(nn.Module):

    def __init__(self, name):
        super().__init__()
        self._name = name
        self._modules = nn.Sequential()

    def append(self, module: nn.Module):
        self._modules.append(module)

    def forward(self, inputs):
        raise NotImplementedError
    
    def cite(self)->str:
        raise NotImplementedError
    
    @property
    def name(self)->str:
        return self._name

class Architechture(nn.Module):
    pass