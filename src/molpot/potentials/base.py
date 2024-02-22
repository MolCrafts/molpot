import torch
import torch.nn as nn
import molpot as mp

class NNPotential:

    def __init__(self, name, *args):
        super().__init__(*args)
        self._name = name
        self._modules = nn.Sequential(*args)
    
    def cite(self)->str:
        raise NotImplementedError
    
    @property
    def name(self)->str:
        return self._name
    
    def forward(self, input):

        input = self._modules(input)
        
        de_drij = torch.autograd.grad(input[mp.alias.energy], input[mp.alias.Rij], grad_outputs=torch.ones_like(input[mp.alias.energy]), retain_graph=True, create_graph=True)[0]

         # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff  
        i_forces = torch.zeros_like(mp.alias.xyz).index_add(0, mp.alias.idx_i, de_drij)
        j_forces = torch.zeros_like(mp.alias.xyz).index_add(0, mp.alias.idx_j, -de_drij)
        input[mp.alias.forces] = i_forces + j_forces
        return input
    
    def grad(self, ):
        return torch.autograd.grad(self._modules,)[0]

class Potential(nn.Module):
    
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._name = name
