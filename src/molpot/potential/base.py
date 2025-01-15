import torch
import torch.nn as nn

class Potential:
    ...
    
class PotentialSeq(Potential, nn.Sequential):
    
    def __init__(self, name, *modules):
        super().__init__(*modules)
        self.name = name
        self.kernel = torch.vmap(self, in_dims=0)
        self.post_process = nn.Sequential()

    def append_post_process(self, module):
        self.post_process.append(module)

    def forward(self, inputs):
        inputs = self.kernel(inputs)
        inputs = self.post_process(inputs)
        return inputs['pred'], inputs['label']
    