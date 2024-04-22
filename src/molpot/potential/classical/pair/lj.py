import torch
import torch.nn as nn
from molpy.potential.pair import LJ126

class LJ126(nn.Module, LJ126):

    def __init__(self, epsilon:float, sigma:float, cutoff:float):
        super().__init__()
        self.register_parameter("epsilon", nn.Parameter(torch.tensor(epsilon)))
        self.register_parameter("sigma", nn.Parameter(torch.tensor(sigma)))
        self.register_parameter("cutoff", nn.Parameter(torch.tensor(cutoff)))

    def forward(self, inputs:dict):
        energy = self.energy(inputs)
        forces = self.forces(inputs)
        inputs["energy"] = energy
        inputs["forces"] = forces
        return inputs
    