'''
High-performance PyTorch operations for neural network potentials
'''
import os.path
import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__), 'libMolPotOpLib.so'))

from .locality.neighbors import get_neighbor_pairs