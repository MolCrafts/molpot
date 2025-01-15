import torch

class LJ126(torch.nn.Module):

    name = "LJ126"

    @staticmethod
    def E(sig, eps, d_ij):
        power_6 = torch.pow(sig / d_ij, 6)
        power_12 = torch.square(power_6)
        return 4 * eps * (power_12 - power_6)

    @staticmethod
    def F(sig, eps, r_ij):
        """Force calculation for Lennard-Jones 12-6 potential"""
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
        power_6 = torch.pow(sig / d_ij, 6)
        power_12 = torch.square(power_6)
        return 24 * eps * (2 * power_12 - power_6) * r_ij / torch.square(d_ij)

    def __init__(self, sig, eps):
        super().__init__()
        self.eps = eps
        self.sig = sig

    def forward(self, inputs, outputs):

        self.energy(inputs, outputs)

        return inputs, outputs

    def energy(self, inputs, outputs):

        d_ij = outputs['pairs'][alias.pair_dist]
        outputs["pairs"]["lj126_energy"] = self.E(self.sig, self.eps, d_ij)

        return inputs, outputs

    def forces(self, inputs, outputs):

        r_ij = outputs["pairs"][alias.pair_diff]
        outputs["pairs"]["lj126_forces"] = self.F(self.sig, self.eps, r_ij)

        return inputs, outputs
