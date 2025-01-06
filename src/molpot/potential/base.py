import torch
import torch.nn as nn
from torch.autograd import grad
from molpot import alias

class Potential:
    ...
    
class PotentialSeq(Potential, nn.Sequential):
    
    def __init__(self, name, *modules):
        super().__init__(*modules)
        self.name = name
        self.kernel = torch.vmap(self, in_dims=0)
        self.post_process = nn.Sequential()

    def forward(self, inputs):
        inputs = self.kernel(inputs)
        return inputs['pred'], inputs['label']
    