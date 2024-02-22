# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-26
# version: 0.0.1

from .base import PairBase
from torch import nn
from molpot import Alias

class LJ126(PairBase):

    def __init__(self, ntypes:int, cutoff:int):
        super().__init__("LJ126", ntypes)

        self.init_params('eps', (ntypes, ntypes))
        self.init_params('sig', (ntypes, ntypes))

    def forward(self, tensor):
        r_ij = tensor[Alias.dist]

        energy = 4 * self.eps * ((self.sig / r_ij)**12 - (self.sig / r_ij)**6)
        tensor[Alias.ti] = energy.sum(-1)
        return tensor