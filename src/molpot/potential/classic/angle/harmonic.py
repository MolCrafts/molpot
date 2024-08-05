from molpot.potential.base import Potential
import torch

class Harmonic(Potential):

    name = "AngleHarmonic"

    def __init__(self, k, theta0):
        super().__init__()
        self.k = k
        self.theta0 = theta0

    def forward(self, inputs, outputs):

        angle = inputs['angle']
        inputs["angle_harmonic_energy"] = 0.5 * self.k * torch.square(angle - self.theta0)

        return inputs, outputs