# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-26
# version: 0.0.1

from ...base import Potential
from torch import nn

class LJ126(Potential):

    def __init__(self, eps, sig):
        super().__init__("LJ126")
        self.eps = nn.Parameter(eps)
        self.sig = nn.Parameter(sig)

    def forward(self, r_ij):
        return 4 * self.eps * ((self.sig / r_ij)**12 - (self.sig / r_ij)**6)