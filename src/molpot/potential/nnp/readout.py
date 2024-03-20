from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from molpot import Alias

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
        scatter_type: str = "add",
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
        self.scatter_type = scatter_type
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, inputs: dict[torch.Tensor]) -> dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs[self.input_key])
        # inputs[self.output_key] = y
        if self.scatter_type == 'add':
            y = torch.squeeze(scatter_add(y, inputs[Alias.idx_m], dim=0, dim_size=torch.max(inputs[Alias.idx_m]) + 1))
        elif self.scatter_type == '':
            pass

        inputs[self.output_key] = torch.squeeze(y)
        return inputs