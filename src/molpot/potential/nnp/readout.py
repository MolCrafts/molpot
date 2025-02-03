from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from molpot import alias

from .block import build_mlp


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """

    in_keys = []
    out_keys = []

    reduce_op = {"sum": torch.index_add}

    def __init__(
        self,
        n_neurons: list[int],
        in_keys: str,
        out_keys: str,
        activation: Callable = F.silu,
        reduce: str = "sum",
        per_atom_output_key: str | None = None,
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
            out_keys: the key under which the result will be stored
            per_atom_output_key: If not None, the key under which the per-atom result will be stored
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.model_outputs = [out_keys]
        self.per_atom_output_key = per_atom_output_key
        if self.per_atom_output_key is not None:
            self.model_outputs.append(self.per_atom_output_key)
        assert len(n_neurons) > 1, ValueError("Need at least one in and one out layer")
        self.n_out = n_neurons[-1]

        self.outnet = build_mlp(
            *n_neurons,
            activation=activation,
        )
        self.reduce = reduce

    def forward(self, *inputs) -> tuple[dict, dict]:
        
        # predict atomwise contributions
        y = self.outnet(inputs[1])  # (n_atoms, n_out)
        if len(inputs) > 1:
            atom_batch = inputs[0]
            result = self.reduce_op[self.reduce](
                torch.zeros(
                    (torch.max(atom_batch) + 1, *y.shape[1:]), device=y.device, dtype=y.dtype
                ),
                0,
                atom_batch,
                y,
            )
        else:
            result = torch.sum(y, dim=0)

        # accumulate the per-atom output if necessary
        if self.per_atom_output_key is not None:
            inputs[self.per_atom_output_key] = y

        return result.squeeze()


class PairForce(nn.Module):

    def __init__(
        self,
        in_keys: str,
        dx_key: str = alias.pair_diff,
        out_keys: str = alias.pair_force,
        create_graph=False,
        retain_graph=True,
    ):
        """
        Derivate `fx_key` w.r.t. `dx_key` and store the result in `out_keys`. `retrain_graph` is set to True if need to compute higher order derivatives or derivate multiple times.

        Args:
            fx_key (str): _description_
            dx_key (str): _description_
            out_keys (str): _description_
            create_graph (bool, optional): _description_. Defaults to False.
            retain_graph (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.in_keys = [in_keys, dx_key, alias.pair_i, alias.pair_j]
        self.out_keys = out_keys
        self.create_graph = create_graph
        self.retain_graph = retain_graph

    def forward(self, *inputs):
        fx, dx, pair_i, pair_j = inputs
        (dfdx,) = torch.autograd.grad(
            torch.sum(fx), dx, create_graph=self.create_graph, retain_graph=True
        )
        atom_force = torch.zeros(
            max(pair_i.max(), pair_j.max()) + 1,
            3,
            dtype=dfdx.dtype,
            device=dfdx.device,
        )
        atom_force = torch.index_add(
            atom_force,
            0,
            pair_i,
            dfdx,
        )
        atom_force = torch.index_add(atom_force, 0, pair_j, dfdx, alpha=-1)
        return atom_force
