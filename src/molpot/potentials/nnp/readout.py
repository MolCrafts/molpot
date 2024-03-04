from typing import Callable, Optional, Sequence
from torch import nn
from torch.nn import functional as F
import torch
from torch_scatter import scatter_add
from molpot import Alias
from .ops import index_add
from .layers import build_mlp

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int = 1,
        n_hidden: Optional[Sequence[int]] = None,
        n_out: int = 1,
        activation: Callable = F.silu,
        aggregate: str = "sum",
        input_key: str = '_T0',
        output_key: str = '_energy',
    ):
        """
        Args:
            n_in: input dimension of representation
            n_out: output dimension of target property (default: 1)
            n_hidden: size of hidden layers.
                If an integer, same number of node is used for all hidden layers resulting
                in a rectangular network.
                If None, the number of neurons is divided by two after each layer starting
                n_in resulting in a pyramidal network.
            n_layers: number of layers.
            aggregation_mode: one of {sum, avg} (default: sum)
            output_key: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super(Atomwise, self).__init__()
        self.n_out = n_out
        if n_hidden is None:
            n_hidden = []
        self.outnet = build_mlp(
            [n_in, *n_hidden, n_out],
            activation=activation,
        )
        self.aggregation = aggregate
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, inputs: dict[torch.Tensor]) -> dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs[self.input_key])
        y = torch.squeeze(y, -1)
        # aggregate
        # if self.aggregation_mode is not None:
        #     idx_m = inputs['_idx_m']
        #     maxm = torch.max(idx_m) + 1
        #     y = index_add(y, 0, idx_m, dim_size=maxm)
        #     if self.aggregation_mode == "avg":
        #         y = y / inputs['_n_atoms']
        idx_m = inputs['_idx_m']
        maxm = torch.max(idx_m) + 1
        y = scatter_add(y, idx_m, dim=0, dim_size=maxm)

        inputs[self.output_key] += y
        return inputs