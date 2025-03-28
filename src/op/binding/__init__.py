"""
High-performance PyTorch operations for neural network potentials
"""

from pathlib import Path
import torch

torch.ops.load_library(Path(__file__).parent/"libmolpot_opLib.so")

from .locality.neighbors import get_neighbor_pairs
from .scatter import scatter_sum, batch_add
from .pot import PMEkernel
