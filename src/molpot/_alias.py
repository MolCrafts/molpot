"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from molpy import Alias
import torch

Alias.set("loss", "_loss", torch.Tensor, None, "loss value")
Alias.set("epoch", "_epoch", int, None, "epoch")