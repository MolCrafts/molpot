import torch
import torch.nn as nn
import molpot as mp

class NNPotential(nn.Sequential):

    def __init__(self, name, *potentials):
        super().__init__(*potentials)
        self._name = name

    def cite(self)->str:
        raise NotImplementedError
    
    @property
    def name(self)->str:
        return self._name
    
    def forward(self, input):

        input[mp.Alias.Rij].requires_grad_(True)
        input = super().forward(input)
        # input[mp.Alias.energy].requires_grad_(True)
        
        de_drij = torch.autograd.grad(input[mp.Alias.energy], input[mp.Alias.Rij], grad_outputs=torch.ones_like(input[mp.Alias.energy]), retain_graph=True, create_graph=True)[0]

         # diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff  
        forces = torch.zeros_like(input[mp.Alias.xyz]).index_add(0, input[mp.Alias.idx_i], de_drij).index_add(0, input[mp.Alias.idx_j], -de_drij)
        input[mp.Alias.forces] = forces
        return input
    
    def grad(self, ):
        return torch.autograd.grad(self._modules,)[0]

class Potential(nn.Module):
    
    def __init__(self, name, *args, **kwargs):
        super().__init__()
        self._name = name
