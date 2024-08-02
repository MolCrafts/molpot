import torch
from molpot.potential.base import Potential

class LJ126(Potential):

    def __init__(self, sig, eps):
        super().__init__('lj126')
        self.eps = eps
        self.sig = sig

    def forward(self, inputs, outputs):

        d_ij = inputs['d_ij']

        power_6 = torch.pow(self.sig / d_ij, 6)
        power_12 = torch.square(power_6)

        outputs['lj126_energy'] = 4 * self.eps * (power_12 - power_6)

        return inputs, outputs