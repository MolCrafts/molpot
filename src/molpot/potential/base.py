import torch
import torch.nn as nn
from torch.autograd import grad

class Potential(nn.Module):

    def __init__(self):
        super().__init__()

    def forces(self, inputs, outputs):
        raise NotImplementedError()
    
class PotentialDict(nn.ModuleDict, Potential):

    pass
    
class PotentialSeq(Potential):

    def __new__(cls, name, *modules):
        cls.name = name
        return super().__new__(cls)
    
    def __init__(self, name, *modules, auto_force=False):
        super().__init__()
        self.potentials = nn.Sequential(*modules)
        self.post_process = nn.Sequential()
        self.auto_force = auto_force

    def forward(self, inputs, outputs):
        for module in self.potentials:
            inputs, outputs = module(inputs, outputs)
        for module in self.post_process:
            inputs, outputs = module(inputs, outputs)
        return inputs, outputs
    
    def __len__(self):
        return len(self.potentials)
    
class CalcForce(nn.Module):

    def __init__(self, energy_key:str = 'energy', force_key:str = 'forces'):
        super().__init__()
        self.energy_key = energy_key
        self.force_key = force_key

    def forward(self, inputs, outputs):

        energy = outputs['atoms'][self.energy_key]
        go = [torch.ones_like(energy)]
        force = grad(energy, inputs['atoms']['xyz'], grad_outputs=go, create_graph=False, retain_graph=True)[0]
        outputs['atoms'][self.force_key] = force
        return inputs, outputs