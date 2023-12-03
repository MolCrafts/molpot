from .metric import *
from .tracker import *

def get_metric(name):
    return globals()[name]