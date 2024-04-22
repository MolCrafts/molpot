from molpy.potential.angle import Harmonic as _Harmonic

class Harmonic(_Harmonic):

    def forward(self, inputs:dict)->dict:
        pos = inputs['_xyz']
        idx_i = inputs['bond_i']
        idx_j = inputs['bond_j']
        energy = self.energy(pos, idx_i, idx_j)
        inputs['_energy'] = energy
        force = self.force(pos, idx_i, idx_j)
        inputs['_force'] = force
        return inputs