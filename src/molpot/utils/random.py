from .frame import Frame
from .box import Box
import torch

def rand_frame(n_atoms:int, size:float|torch.Tensor=10):

    box = Box(size)
    xyz = torch.rand((n_atoms, 3)) * box.lengths
    frame = Frame()
    frame['atoms']['xyz'] = xyz
    frame['box']['cell'] = box.matrix
    frame['box']['pbc'] = torch.tensor([True, True, True])

    return frame
