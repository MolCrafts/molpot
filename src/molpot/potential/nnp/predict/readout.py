from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from molpot import alias
from ..base import FeedForward
from molpot_op.scatter import batch_add, get_natoms_per_batch

from torch.nn import LazyLinear


class Batchwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.
    """
    reduce_op = {"sum": torch.index_add}

    def __init__(
        self,
        n_neurons: list[int],
        in_key: str,
        out_key: str,
        activation: Callable = F.silu,
        reduce: str = "sum"
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
        self.in_keys = [alias.atom_batch, in_key]
        self.out_keys = out_key if not isinstance(out_key, str) else ("predicts", out_key)
        assert len(n_neurons) > 1, ValueError("Need at least one in and one out layer")
        self.n_out = n_neurons[-1]

        self.outnet = FeedForward(
            *n_neurons,
            activation=activation,
            last_bias=False,
        )
        self.reduce = reduce

    def forward(self, atom_batch, px) -> tuple[dict, dict]:

        y = self.outnet(px)  # (n_atoms, n_out)
        result = batch_add(y, atom_batch)
        return result.squeeze()


class PairForce(nn.Module):

    def __init__(
        self,
        in_key: str,
        out_key: str,
        dx_key: str = alias.pair_diff,
        create_graph=True,
        retain_graph=True,
    ):
        """
        Derivate `fx_key` w.r.t. `dx_key` and store the result in `out_keys`. `retrain_graph` is set to True if need to compute higher order derivatives or derivate multiple times. `create_graph` is set to True if force is included in loss.

        Args:
            fx_key (str): _description_
            dx_key (str): _description_
            out_keys (str): _description_
            create_graph (bool, optional): _description_. Defaults to False.
            retain_graph (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.in_keys = [in_key, dx_key, alias.pair_i, alias.pair_j]
        self.out_keys = ("predicts", out_key)
        self.create_graph = create_graph
        self.retain_graph = retain_graph

    def forward(self, fx, dx, pair_i, pair_j):
        dfdx = -1 * torch.autograd.grad(
            torch.sum(fx),
            dx,
            create_graph=self.create_graph,
            retain_graph=self.retain_graph,
        )[0]
        atom_force = torch.zeros(
            max(pair_i.max(), pair_j.max()) + 1,
            3,
            dtype=dfdx.dtype,
            device=dfdx.device,
        )
        atom_force = torch.index_add(
            atom_force,
            0,
            pair_j,
            dfdx,
        )
        atom_force = torch.index_add(atom_force, 0, pair_i, dfdx, alpha=-1)
        return atom_force


class SystemChargeNeutralize(nn.Module):

    def __init__(self, in_keys: str, out_keys: str):
        self.in_keys = [alias.atom_batch, in_keys]
        self.out_keys = out_keys
        super().__init__()
        self.layer = LazyLinear(1)

    def forward(self, atom_batch, p1):
        p1 = self.layer(p1).squeeze()
        q_batch = batch_add(p1, atom_batch)  # shape (n_batch,)
        natoms_per_molecule = get_natoms_per_batch(atom_batch)
        p_charge = q_batch / natoms_per_molecule
        charge_corr = p_charge[atom_batch]
        return (p1 - charge_corr).reshape(-1, 1)

class DipoleAC(nn.Module):

    def __init__(self, in_keys: str, out_keys: str, return_norm: bool = False):
        self.in_keys = [alias.atom_batch, alias.xyz, in_keys]
        self.out_keys = out_keys
        super().__init__()
        self.layer = LazyLinear(1)
        self.return_norm = return_norm

    def forward(self, atom_batch, xyz, p1: torch.Tensor):
        p1 = self.layer(p1)
        q_d = p1 * xyz
        dipole = batch_add(q_d, atom_batch)
        if self.return_norm:
            dipole_norm = torch.norm(dipole, dim=-1)
            return dipole, dipole_norm
        return dipole