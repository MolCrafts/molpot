import torch
from torch import nn
from .block import build_mlp
from typing import Callable
from torch.functional import F

class FFLayer(nn.Module):

    def __init__(self, n_nodes:list[int]=[64, 64], act:Callable=F.tanh):

        super().__init__()

        n_in, *n_hidden, n_out = n_nodes
        self.dense_layers = build_mlp(n_in, n_out, n_hidden, n_layers=len(n_nodes), activation=nn.ReLU)

    def forward(self, tensors:torch.Tensor):

        return self.dense_layers(tensors)

# class PILayer(nn.Module):

#     def __init__(self, n_nodes=[64], ):

#         super().__init__()
#         self.n_nodes = n_nodes
#         self.layers = 

#     def forward(self, tensors:torch.Tensor):

