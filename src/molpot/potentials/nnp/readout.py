from typing import Callable, Dict, Optional, Sequence, Union
from torch import nn
from torch.nn import functional as F
import torch
from molpot import alias
from .ops import scatter_add
from .layers import Dense

def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    last_bias: bool = True,
    last_zero_init: bool = False,
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation)
        for i in range(n_layers - 1)
    ]
    # assign a Dense layer (without activation function) to the output layer

    if last_zero_init:
        layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=last_bias,
            )
        )
    else:
        layers.append(
            Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net

class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = "y",
        per_atom_output_key: Optional[str] = None,
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
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        # aggregate
        if self.aggregation_mode is not None:
            idx_m = inputs[alias.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                y = y / inputs[alias.n_atoms]

        inputs[self.output_key] = y
        return inputs