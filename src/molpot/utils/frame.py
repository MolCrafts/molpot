
class Frame(dict):

    def __init__(self):

        self['atoms'] = {}
        self['bonds'] = {}
        self['angles'] = {}
        self['dihedrals'] = {}
        self['impropers'] = {}
        self['pairs'] = {}
        self['box'] = {}