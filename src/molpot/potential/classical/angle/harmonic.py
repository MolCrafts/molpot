from molpy.potential.angle import Harmonic as _Harmonic

class Harmonic(_Harmonic):

    def forward(self, inputs:dict)->dict:
        pos = inputs['_xyz']
        idx_i = inputs['angle_i']
        idx_j = inputs['angle_j']
        idx_k = inputs['angle_k']
        energy = self.energy(pos, idx_i, idx_j, idx_k)
        inputs['_energy'] = energy
        force = self.force(pos, idx_i, idx_j, idx_k)
        inputs['_force'] = force
        return inputs