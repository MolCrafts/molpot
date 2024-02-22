"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from molpy import Alias
import torch

Alias.set("T0", "_T0", torch.Tensor, None, "rank 0 tensor")
Alias.set("T1", "_T1", torch.Tensor, None, "rank 1 tensor")
Alias.set("T2", "_T2", torch.Tensor, None, "rank 2 tensor")

Alias.set("loss", "_loss", torch.Tensor, None, "loss value")
Alias.set("epoch", "_epoch", int, None, "epoch")