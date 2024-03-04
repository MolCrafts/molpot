# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-12
# version: 0.0.1

from .readout import Atomwise
from ..base import Potential
import torch
from torch import nn
from .layers import Dense, build_mlp
import molpot as mpot
from molpot import Alias
from typing import Optional, Callable, Sequence
import torch.nn.functional as F
from .ops import index_add
from torch_scatter import scatter_add


class PILayer(nn.Module):
    def __init__(self, n_channel: int, n_hidden: Sequence[int], n_basis: int, **kwargs):
        super().__init__()
        self.n_basis = n_basis
        self.n_neurons = [n_channel * 2, *n_hidden, n_hidden[-1] * n_basis]

        self.ff_layer = build_mlp(
            self.n_neurons, activation=None, last_bias=False, **kwargs
        )

    def forward(self, prop, idx_i, idx_j, basis):
        prop_i = prop[idx_i]
        prop_j = prop[idx_j]  # (n_pairs, n_channel)
        inter = torch.cat([prop_i, prop_j], axis=-1)
        inter = self.ff_layer(inter)  # NOTE: weight? (n_pairs, n_hidden[-1] * n_basis)
        inter = inter.reshape(
            [*inter.shape[:-1], self.n_neurons[-2], self.n_basis]
        )  # (n_pairs, n_hidden[-1], n_basis)
        inter = torch.einsum("i...c,ic->i...", inter, basis)
        return inter


class DotLayer(nn.Module):

    def forward(self, px):
        return torch.einsum("i...c,i...c->ic", px, px)


class ScaleLayer(nn.Module):

    def forward(self, px, p1):
        return torch.einsum("i...c,i...c->i...c", px, p1)


class AddLayer(nn.Module):

    @staticmethod
    def add_i3(i3_, i3):
        return i3_ + i3

    @staticmethod
    def add_i5(i5, i3):
        vvt = torch.einsum("ijl, ikl->ijkl", i3, i3)
        vvt_trace = torch.einsum("ijjl->il", vvt)  # [i,3,3,j] -> [i,j]
        vvt_shape_id = (
            torch.eye(3, device=i3.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand((vvt_trace.shape[0], 3, 3, vvt_trace.shape[-1]))
        )
        vvt_trace_diag = vvt_trace[:, None, None, :] * vvt_shape_id / 3
        S = vvt - vvt_trace_diag
        return i5 + S

    def __init__(self, n_dim: int):
        super().__init__()
        if n_dim == 3:
            self.add = AddLayer.add_i3
        elif n_dim == 5:
            self.add = AddLayer.add_i5
        else:
            raise NotImplementedError("only rank 2 and rank 3 allowed")

    def forward(self, ix, i3):
        return self.add(ix, i3)


class IPLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i, idx_i):
        maxm = torch.max(idx_i) + 1
        # p = torch.zeros(maxm, *i.shape[1:], device=i.device, requires_grad=True)
        # p = p.index_add(0, idx_i, i)
        p = scatter_add(i, idx_i, dim=0, dim_size=maxm)
        return p


class OutLayer(nn.Module):
    def __init__(self, n_channel: int, n_hidden: Sequence[int], out_units: int):
        super().__init__()

        self.ff_layer = build_mlp([n_channel, *n_hidden, out_units])

    def forward(self, p):
        return self.ff_layer(p)


class GCBlockP1(nn.Module):

    def __init__(self, pp_nodes, pi_nodes, ii_nodes, n_basis, activation):
        super().__init__()
        self.pp_layer = build_mlp(pp_nodes, activation)
        self.pi_layer = PILayer(pp_nodes[-1], pi_nodes, n_basis)
        self.ii_layer = build_mlp(ii_nodes, activation)
        self.ip_layer = IPLayer()

    def forward(self, p, idx_i, idx_j, basis):
        p1 = self.pp_layer(p)
        i1 = self.pi_layer(p1, idx_i, idx_j, basis)
        i1 = self.ii_layer(i1)
        p = self.ip_layer(i1, idx_i)
        return p


class GCBlockP3(nn.Module):
    def __init__(self, pp_nodes, pi_nodes, ii_nodes, n_basis, activation):
        super().__init__()

        pi1_nodes = pi_nodes.copy()
        pi1_nodes[-1] = pi1_nodes[-1] * 3
        pi3_nodes = pi_nodes.copy()

        self.pp1_layer = build_mlp(pp_nodes, activation)
        self.pi1_layer = PILayer(pp_nodes[-1], pi1_nodes, n_basis)
        self.ii1_layer = build_mlp(ii_nodes, activation)
        self.ip1_layer = IPLayer()

        self.pp3_layer = build_mlp(pp_nodes, activation)
        self.pi3_layer = PILayer(pp_nodes[-1], pi3_nodes, n_basis)
        self.ii3_layer = build_mlp(ii_nodes, activation)
        self.ip3_layer = IPLayer()

        self.dot_layer = DotLayer()

        self.i1i3_scale_layer = ScaleLayer()
        self.i1r3_scale_layer = ScaleLayer()
        self.p1p3_scale_layer = ScaleLayer()

        self.i3_add_layer = AddLayer(n_dim=3)

    def forward(self, p1, p3, r3, idx_i, idx_j, basis):

        p1 = self.pp1_layer(p1)
        i1 = self.pi1_layer(p1, idx_i, idx_j, basis)
        i1_1, i1_2, i1_3 = torch.split(i1, int(i1.shape[-1] / 3), dim=-1)
        i1 = self.ii1_layer(i1_1)
        p1 = self.ip1_layer(i1, idx_i)

        p3 = self.pp3_layer(p3)
        i3 = self.pi3_layer(p3, idx_i, idx_j, basis)
        i3 = self.ii3_layer(i3)
        i3 = self.i1i3_scale_layer(i3, i1_2)
        scaled_r3 = self.i1r3_scale_layer(r3[..., None], i1_3)

        i3 = self.i3_add_layer(i3, scaled_r3)
        p3 = self.ip3_layer(i3, idx_i)

        p1 = self.dot_layer(p3) + p1
        p3 = self.p1p3_scale_layer(p3, p1)

        return p1, p3


class PiNet(Potential):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        n_atom_basis: int,
        depth: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
    ):

        super().__init__("PiNet")
        if activation == "tanh":
            activation = nn.Tanh()
        self.depth = depth
        mpot.Alias("pinet")
        Alias.pinet.set("p1", "_pinet_p1", torch.Tensor, None, "invariant property")
        self.cutoff_fn = cutoff_fn
        self.radial_basis_fn = radial_basis
        self.n_basis = radial_basis.n_basis
        self.n_atom_basis = n_atom_basis
        self.embbding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        self.gc_blocks = nn.Sequential(
            *[
                GCBlockP1(
                    pp_nodes, pi_nodes, ii_nodes, self.n_basis, activation=activation
                )
                for _ in range(depth)
            ]
        )
        self.out_layers = [OutLayer(n_atom_basis, out_nodes, 1) for _ in range(depth)]

    def forward(self, inputs: dict):

        R = inputs[Alias.R]
        idx_i = inputs[Alias.idx_i]
        idx_j = inputs[Alias.idx_j]
        offsets = inputs[Alias.offsets]
        r_ij = R[idx_i] - R[idx_j] + offsets
        d_ij = torch.norm(r_ij, dim=-1)
        p1 = torch.squeeze(inputs[Alias.Z])
        p1 = self.embbding(p1)
        fc = self.cutoff_fn(d_ij)
        basis = self.radial_basis_fn(d_ij, fc)
        # output = 0.0  # broadcast to shape:= (n_atoms, 1)
        for i in range(self.depth):

            p1 = self.gc_blocks[i](
                p1,
                idx_i,
                idx_j,
                basis,
            )
            # output = self.out_layers[i](p1, output)

        inputs[Alias.pinet.p1] = p1
        return inputs


class PiNetP3(Potential):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        n_atom_basis: int,
        depth: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
    ):

        super().__init__("PiNetP3")
        if activation == "tanh":
            activation = nn.Tanh()
        self.depth = depth
        mpot.Alias("pinet")
        Alias.pinet.set("p1", "_pinet_p1", torch.Tensor, None, "invariant property")
        Alias.pinet.set(
            "p3", "_pinet_p3", torch.Tensor, None, "rank1 equivalent property"
        )
        Alias.pinet.set(
            "p5", "_pinet_p5", torch.Tensor, None, "rank2 equivalent property"
        )
        Alias.pinet.set(
            "output_p1", "_pinet_output_p1", torch.Tensor, None, "output property"
        )
        self.cutoff_fn = cutoff_fn
        self.radial_basis_fn = radial_basis
        self.n_basis = radial_basis.n_basis
        self.n_atom_basis = n_atom_basis
        self.embbding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        self.gc_blocks = nn.Sequential(
            *[
                GCBlockP3(
                    pp_nodes, pi_nodes, ii_nodes, self.n_basis, activation=activation
                )
                for _ in range(depth)
            ]
        )
        self.out_layers = [OutLayer(n_atom_basis, out_nodes, 1) for _ in range(depth)]
        # self.ann_output = Atomwise(1, [], 1, aggregation_mode=out_pool)

    def forward(self, inputs: dict):

        R = inputs[Alias.R]
        idx_i = inputs[Alias.idx_i]
        idx_j = inputs[Alias.idx_j]
        offsets = inputs[Alias.offsets]
        offsets = offsets.requires_grad_(True)
        r_ij = R[idx_i] - R[idx_j] + offsets
        d_ij = torch.norm(r_ij, dim=-1)
        p1 = torch.squeeze(inputs[Alias.Z])
        p1 = self.embbding(p1)
        p3 = torch.zeros(p1.shape[0], 3, p1.shape[-1], requires_grad=True)
        fc = self.cutoff_fn(d_ij)
        basis = self.radial_basis_fn(d_ij, fc)
        # output = 0.0  # broadcast to shape:= (n_atoms, 1)
        for i in range(self.depth):
            p1, p3 = self.gc_blocks[i](
                p1,
                p3,
                r_ij,
                idx_i,
                idx_j,
                basis,
            )
            print(p1)
            # output += self.out_layers[i](p1)

        inputs[Alias.pinet.p1] = p1
        inputs[Alias.pinet.p3] = p3
        return inputs
