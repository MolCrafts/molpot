import importlib
from .pair import *
from .bond import *
from .angle import *

def get_classic_potental(name, style):
    pair_module = importlib.import_module(f".{style}", __package__)
    return getattr(pair_module, name)
