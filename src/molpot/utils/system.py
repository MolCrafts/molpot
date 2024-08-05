
class System:

    def __init__(self):

        self._frame = None
        self._forcefield = None

    @property
    def frame(self):
        return self._frame
    
    @frame.setter
    def frame(self, frame):
        self._frame = frame

    @property
    def forcefield(self):
        return self._forcefield
    
    @forcefield.setter
    def forcefield(self, forcefield):
        self._forcefield = forcefield

    @property
    def potential(self):
        return self.forcefield.get_potential()
    