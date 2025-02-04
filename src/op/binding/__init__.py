'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libmolpot_opLib.so'))

from .locality.neighbors import get_neighbor_pairs
from .scatter import scatter_add, scatter_sum
from .pot import PME, PMEkernel