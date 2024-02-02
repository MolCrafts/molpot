from ...base import Potential
from torch import nn
import torch

class PairBase(Potential):

    def __init__(self, name, ntypes:int):
        super().__init__(name)
        self.ntypes = ntypes

    def init_params(self, name:str, shape:tuple):
        params = nn.Parameter(torch.zeros(shape))
        self.register_parameter(name, params)
        return params
