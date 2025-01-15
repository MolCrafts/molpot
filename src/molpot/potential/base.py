import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule

class PotentialSeq(nn.Module):

    def __init__(self, *modules):
        super().__init__()
        td_modules = [TensorDictModule(module, in_keys=module.in_keys, out_keys=module.out_keys) for module in modules]
        self.kernel = TensorDictSequential(*td_modules)
        self.derivative = None

    def forward(self, inputs):
        inputs = torch.vmap(self.kernel, in_dims=(0, ))(inputs)
        if self.derivative is not None:
            inputs = self.derivative(inputs)

        return inputs["predicts"], inputs["labels"]
