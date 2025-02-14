import torch.nn as nn

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

    def forward(self, inputs):
        return self.kernel(inputs)
    