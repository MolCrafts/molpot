from abc import abstractmethod
from typing import Sequence
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

    in_keys: tuple[str, ...]
    out_keys: str

    def __init__(self, out_keys: str, in_keys: Sequence[str], reduce = torch.sum):
        super().__init__()
        self.out_keys = ("predicts", out_keys)
        self.in_keys = [("predicts", key) for key in in_keys]
        self.reduce = reduce

    def forward(self, *args):
        return self.reduce(args, dim=0)
