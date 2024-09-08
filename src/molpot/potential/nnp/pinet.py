import torch
import torch.nn as nn
import torch.nn.functional as F
from molpot import NameSpace
from .block import build_mlp
from typing import Callable
from molpot import alias
from molpot_op.scatter import scatter_add


class PPLayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list[int] = [],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.mlp = build_mlp(
            n_in, n_out, n_hidden, use_bias=False, activation=activation
        )

    def forward(self, prop):
        return self.mlp(prop)


class IILayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list[int] = [],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.mlp = build_mlp(
            n_in, n_out, n_hidden, use_bias=False, activation=activation
        )

    def forward(self, prop):
        return self.mlp(prop)


class PILayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list[int],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.mlp = build_mlp(
            n_in,
            n_out,
            n_hidden,
            use_bias=False,
            activation=activation,
        )

    def forward(self, prop, idx_i, idx_j, basis):

        prop_i = prop[idx_i]
        prop_j = prop[idx_j]

        inter = prop_i + prop_j
        inter = self.mlp(inter)

        inter = torch.einsum(
            "pcb, pb->pc", inter.reshape(-1, prop_i.shape[-1], basis.shape[-1]), basis
        )  # (n_atoms, n_channels)
        return inter[:, None, :]  # (n_atoms, 1, n_channels)


class PIXLayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list[int] = [],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.wi = build_mlp(
            n_in,
            n_out,
            n_hidden,
            use_bias=False,
            activation=activation,
        )
        self.wj = build_mlp(
            n_in,
            n_out,
            n_hidden,
            use_bias=False,
            activation=activation,
        )

    def forward(self, prop, idx_i, idx_j):

        prop_i = prop[idx_i]
        prop_j = prop[idx_j]

        return self.wi(prop_i) + self.wj(prop_j)


class IPLayer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, idx_i, inter):

        return scatter_add(inter, idx_i, dim=0)

class InvarLayer(nn.Module):

    def __init__(
        self,
        pp_nodes: list[int],
        pi_nodes: list[int],
        ii_nodes: list[int],
        activation: Callable = F.tanh,
    ):
        super().__init__()
        self.pp_layer = PPLayer(pp_nodes[0], pp_nodes[-1], pp_nodes[1:-1], activation)
        self.pi_layer = PILayer(pi_nodes[0], pi_nodes[-1], pi_nodes[1:-1], activation)
        self.ii_layer = IILayer(ii_nodes[0], ii_nodes[-1], ii_nodes[1:-1], activation)
        self.ip_layer = IPLayer()

    def forward(self, idx_i, idx_j, p1, basis):

        i1 = self.pi_layer(p1, idx_i, idx_j, basis)
        i1 = self.ii_layer(i1)
        p1 = self.ip_layer(idx_i, i1)
        p1 = self.pp_layer(p1)

        return p1, i1


class EqvarLayer(nn.Module):

    def __init__(self, n_nodes: list[int]):
        super().__init__()
        self.pp_layer = PPLayer(n_nodes[0], n_nodes[-1], activation=None)
        self.pi_layer = PIXLayer(n_nodes[0], n_nodes[-1], activation=None)
        self.ii_layer = IILayer(n_nodes[0], n_nodes[-1], activation=None)
        self.ip_layer = IPLayer()

        self.scale_layer = ScaleLayer()

    def forward(self, idx_i, idx_j, px, diff, i1):

        ix = self.pi_layer(px, idx_i, idx_j)
        ix = self.scale_layer(ix, i1)
        scaled_diff = self.scale_layer(diff[:, :, None], i1)
        ix = ix + scaled_diff
        px = self.ip_layer(idx_i, ix)
        px = self.pp_layer(px)
        ix = self.ii_layer(ix)

        return px, ix


class ScaleLayer(nn.Module):

    def forward(self, px, p1):
        return px * p1


class SelfDotLayer(nn.Module):

    def forward(self, p):
        return torch.einsum("ixr, ixr->ir", p, p)


class GCBlock(nn.Module):

    def __init__(
        self,
        pp_nodes: list[int],
        pi_nodes: list[int],
        ii_nodes: list[int],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.p1_layer = InvarLayer(pp_nodes, pi_nodes, ii_nodes, activation)
        self.p3_layer = EqvarLayer([pp_nodes[0], pp_nodes[-1]])

        self.scale_layer = ScaleLayer()
        self.dot_layer = SelfDotLayer()

    def forward(self, inputs) -> dict[str, torch.Tensor]:
        pair_i = inputs[alias.pair_i]
        pair_j = inputs[alias.pair_j]
        basis = inputs["pinet", "basis"]
        p1 = inputs["pinet", "p1"]
        p1, i1 = self.p1_layer(pair_i, pair_j, p1, basis)
        inputs["pinet", "p1"] = p1
        inputs["pinet", "i1"] = i1

        p3 = inputs["pinet", "p3"]
        diff_p3 = inputs['pairs', 'norm_diff'] 
        p3, i3 = self.p3_layer(pair_i, pair_j, p3, diff_p3, i1)
        inputs["pinet", "p3"] = p3
        inputs["pinet", "i3"] = i3
        return inputs


class ResUpdate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, old, new):
        return old + new


class PiNet(nn.Module):

    def __init__(
        self,
        depth: int,
        basis_fn: Callable | None = None,
        cutoff_fn: Callable | None = None,
        pp_nodes: int = [16, 16],
        pi_nodes: int = [16, 16],
        ii_nodes: int = [16, 16],
        activation: Callable | None = F.tanh,
        max_atomtypes: int = 100
    ):
        super().__init__()

        self.labels = NameSpace("pinet")
        self.labels.set(
            "p1",
            torch.Tensor,
            "unit",
            "scalar property",
            ("n_atoms", "n_components", "n_channels"),
        )
        self.labels.set(
            "p3",
            torch.Tensor,
            "unit",
            "vectorial property",
            ("n_atoms", "n_components", "n_channels"),
        )

        self.depth = depth
        self.basis_fn = basis_fn
        self.cutoff_fn = cutoff_fn
        self.n_basis = self.basis_fn.n_rbf

        self.embedding = nn.Embedding(max_atomtypes, max_atomtypes, padding_idx=0)

        pp_nodes = [ii_nodes[-1], *pp_nodes]
        pi_nodes = [pp_nodes[-1], *pi_nodes]
        ii_nodes = [pi_nodes[-1], *ii_nodes]

        pi_nodes[-1] *= self.n_basis

        self.before_gc_block_layer = nn.Linear(max_atomtypes, pp_nodes[0])

        self.gc_blocks = nn.ModuleList(
            [
                GCBlock(pp_nodes, pi_nodes, ii_nodes, activation)
                for _ in range(depth)
            ]
        )

        self.res_update = ResUpdate()

    def forward(self, inputs: dict[str, torch.Tensor]) -> None:

        # get tensors from input dictionary
        Z = inputs[alias.Z]
        n_atoms = inputs[alias.n_atoms]
        r_ij = inputs[alias.pair_diff]
        d_ij = inputs[alias.pair_dist]
        r_ij /= torch.norm(r_ij, dim=-1, keepdim=True)
        inputs['pairs', 'norm_diff'] = r_ij

        basis = self.basis_fn(d_ij)
        fc = self.cutoff_fn(d_ij)

        inputs["pinet", "basis"] = basis * fc[..., None]
        p1 = self.embedding(Z)[:, None, :]

        p1 = self.before_gc_block_layer(p1)
        p3 = torch.zeros([n_atoms, 3, p1.shape[-1]], device=p1.device)

        inputs["pinet", "p1"] = p1
        inputs["pinet", "p3"] = p3

        for i in range(self.depth):
            inputs = self.gc_blocks[i](inputs)
            inputs["pinet", "p1"] = self.res_update(inputs["pinet", "p1"], p1)
            inputs["pinet", "p3"] = self.res_update(inputs["pinet", "p3"], p3)
            p1 = inputs["pinet", "p1"]
            p3 = inputs["pinet", "p3"]
        return inputs
