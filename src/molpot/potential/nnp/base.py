from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, zeros_


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Callable | nn.Module | None = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class FeedForward(nn.Module):
    def __init__(
        self,
        *n_nodes: int,
        bias: bool = True,
        activation: Callable | nn.Module | None = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
        last_bias: bool = True,
        last_zero_init: bool = False,
    ):
        super().__init__()

        layers = [
            Dense(
                n_nodes[i],
                n_nodes[i + 1],
                bias=bias,
                activation=activation,
                weight_init=weight_init,
                bias_init=bias_init,
            )
            for i in range(len(n_nodes) - 2)
        ]

        if last_zero_init:
            layers.append(
                Dense(
                    n_nodes[-2],
                    n_nodes[-1],
                    activation=None,
                    weight_init=torch.nn.init.zeros_,
                    bias=last_bias,
                )
            )
        else:
            layers.append(
                Dense(n_nodes[-2], n_nodes[-1], activation=None, bias=last_bias)
            )
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)