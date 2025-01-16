import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule


class PotentialSeq(nn.Module):

    def __init__(self, *modules):
        super().__init__()
        self.kernel = TensorDictSequential(
            *[
                TensorDictModule(
                    module, in_keys=module.in_keys, out_keys=module.out_keys
                )
                for module in modules
            ]
        )
        self.vmapped_kernel = torch.vmap(self.kernel, in_dims=0)
        self.derivative = None

    def forward(self, inputs):
        # assert inputs.batch_size == (1, )
        inputs = self.vmapped_kernel(inputs)
        if self.derivative is not None:
            inputs = self.derivative(inputs)

        return inputs