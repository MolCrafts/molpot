import torch
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

    def forward(self, td):
        return self.kernel(td)
    
class Reducer(nn.Module):

    def __init__(self, target: str, *keys: str):
        super().__init__()
        self.target = self.out_keys = ("predicts", target)
        self.keys = self.in_keys = [("predicts", key) for key in keys]

    def forward(self, td):
        td[self.target] = torch.stack([td[key] for key in self.keys], dim=0)
        return td