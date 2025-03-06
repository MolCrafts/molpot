from abc import abstractmethod
import torch
import torch.nn as nn

from tensordict.nn import TensorDictSequential, TensorDictModuleBase, TensorDictModule


class Potential(TensorDictModuleBase):

    def cite(self):
        return f"{self.__class__.__name__} from molpot"


class PotentialSeq(TensorDictSequential):

    def __init__(self, *modules):
        super().__init__(
            *[
                TensorDictModule(
                    module, in_keys=module.in_keys, out_keys=module.out_keys
                )
                for module in modules
            ]
        )


class Reducer(nn.Module):

    def __init__(self, target: str, *keys: str):
        super().__init__()
        self.target = self.out_keys = ("predicts", target)
        self.keys = self.in_keys = [("predicts", key) for key in keys]

    def forward(self, td):
        td[self.target] = torch.stack([td[key] for key in self.keys], dim=0)
        return td
