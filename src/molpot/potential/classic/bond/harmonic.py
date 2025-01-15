import torch

class  Harmonic(torch.nn.Module):

    name = "BondHarmonic"

    def __init__(self, r0, k):
        super().__init__()
        self.r0 = r0
        self.k = k

    def forward(self, inputs):
        
        typ = inputs['atoms', 'type']
        bondtype_i = typ[inputs['bonds', 'i']]
        bondtype_j = typ[inputs['bonds', 'j']]

        k = self.k[bondtype_i, bondtype_j]
        r0 = self.r0[bondtype_i, bondtype_j]

        bond_dist = inputs['bonds', "dist"]
        inputs["predicts", "bond_harmonic_energy"] = 0.5 * k * torch.square(bond_dist - r0)

        return inputs
    