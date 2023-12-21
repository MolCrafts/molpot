import torch
import torch.nn as nn

class NNPotential(nn.Module):

    def __init__(self, name):
        super().__init__()
        self._name = name
        self._layers = nn.Sequential()

    def append(self, module: nn.Module):
        self._layers.append(module)

    def forward(self, inputs):
        return self._layers(inputs)
    
    def cite(self)->str:
        raise NotImplementedError
    
    @property
    def name(self)->str:
        return self._name

class Architechture(nn.Module):
    pass