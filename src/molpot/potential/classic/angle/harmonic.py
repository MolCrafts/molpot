from molpot.potential.base import Potential
import torch

class Hamonic(Potential):

    def __init__(self, r0, theta):
        super().__init__('AngleHarmonic')
        self.r0 = r0
        self.theta = theta

    def forward(self, inputs, outputs):

        angle = inputs['angle']
        inputs["angle_harmonic_energy"] = 0.5 * self.theta * torch.square(angle - self.theta)

        return inputs, outputs