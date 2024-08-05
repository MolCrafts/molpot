import torch
from molpot.potential.base import Potential

class Harmonic(Potential):

    name = "BondHarmonic"

    def __init__(self, r0, k):
        super().__init__()
        self.r0 = r0
        self.k = k

    def forward(self, inputs, outputs):

        bond_dist = inputs['bond_dist']
        inputs["bond_harmonic_energy"] = 0.5 * self.k * torch.square(bond_dist - self.r0)

        return inputs, outputs
    