# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-26
# version: 0.0.1

from molpot.potentials.base import Potential

class LJ126(Potential):

    def __init__(self, eps, sig, r_cut, r_switch, map_prm, map_nbfix):
        super().__init__("LJ126")
        self.eps = eps
        self.sig = sig
        self.r_cut = r_cut
        self.r_switch = r_switch

    def forward(self):
        pass