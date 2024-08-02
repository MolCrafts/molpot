
class Frame:

    def __init__(self):

        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}

    def get_state(self):

        return {
            'atoms': self.atoms,
            'bonds': self.bonds,
            'angles': self.angles,
            'dihedrals': self.dihedrals,
            'impropers': self.impropers
        }