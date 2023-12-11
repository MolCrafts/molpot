import torch
import torch.nn as nn

class Potential(nn.Module):

    def __init__(self, name):
        super().__init__()
        self._name = name

    def forward(self, inputs):
        raise NotImplementedError
    
    def cite(self)->str:
        raise NotImplementedError
    
    @property
    def name(self)->str:
        return self._name
