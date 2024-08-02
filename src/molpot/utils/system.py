
class System:

    def __init__(self):

        self.frame = None
        self.forcefield = None

    @property
    def potential(self):
        return self.forcefield.get_potential()
    
    @property
    def inputs(self):
        inputs = self.frame.get_state()

        return inputs