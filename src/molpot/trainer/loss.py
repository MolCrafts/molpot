# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-12
# version: 0.0.1

from torch import nn
import molpot as mpot

class MSELoss(nn.Module):

    def __init__(self, e_weight:int=1, f_weight:int=1, s_weight:int=1):
        super().__init__()
        self.e_weight = e_weight
        self.f_weight = f_weight
        self.s_weight = s_weight

    def forward(self, input, target):
        te = target[mpot.energy]
        tf = target[mpot.forces]
        ts = target[mpot.stress]
        ie = input[mpot.energy]
