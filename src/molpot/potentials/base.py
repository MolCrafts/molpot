import torch
import torch.nn as nn

class NNPotential(nn.Sequential):

    def __init__(self, name, *args):
        super().__init__(*args)
        self._name = name
    
    def cite(self)->str:
        raise NotImplementedError
    
    @property
    def name(self)->str:
        return self._name

class Potential(nn.Module):
    
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._name = name

class Potentials(nn.Sequential):
    pass