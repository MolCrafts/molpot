import torch
import torch.nn as nn

class Potential(nn.Module):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, inputs):
        raise NotImplementedError
