from .nnp import *
from .classical import *
from .base import NNPotential

def get_potential(config:dict)->NNPotential:
    potential_name = config['type']
    potential_class = globals()[potential_name]
    potential_ins = potential_class(**config['args'])

    return potential_ins