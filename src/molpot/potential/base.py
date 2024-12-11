import torch
import torch.nn as nn
from torch.autograd import grad
from molpot import alias

class Potential(nn.Module):

    def __init__(self):
        super().__init__()

    def forces(self, inputs):
        raise NotImplementedError()
    
class PotentialDict(nn.ModuleDict, Potential):

    pass
    
class PotentialSeq(Potential):

    def __new__(cls, name, *modules):
        assert isinstance(name, str), "name must be a string"
        cls.name = name
        return super().__new__(cls)
    
    def __init__(self, name, *modules):
        super().__init__()
        self.potentials = nn.Sequential(*modules)
        self.post_process = nn.Sequential()

    def forward(self, inputs):
        for module in self.potentials:
            inputs = module(inputs)
        for module in self.post_process:
            inputs = module(inputs)
        return inputs['pred'], inputs['label']
    
    def __len__(self):
        return len(self.potentials)
    
class CalcForce(nn.Module):

    def __init__(self, energy_key:str = 'energy', force_key:str = 'forces'):
        super().__init__()
        self.energy_key = energy_key
        self.force_key = force_key

    def forward(self, inputs):

        energy = inputs[self.energy_key]
        go = [torch.ones_like(energy)]
        force = grad(energy, inputs[alias.xyz], grad_outputs=go, create_graph=False, retain_graph=True)[0]
        inputs[self.force_key] = force
        return inputs