import torch
import torch.nn as nn
import torch.nn.functional as F
from molpot import NameSpace
from .block import build_mlp
from typing import Callable
from torch_scatter import scatter_add
from molpot import alias


class PPLayer(nn.Module):

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: list[int],
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
        n_hidden: list[int],
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
        n_hidden: list[int],
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

        p1 = self.pp_layer(p1)
        i1 = self.pi_layer(p1, idx_i, idx_j, basis)
        i1 = self.ii_layer(i1)
        p1 = self.ip_layer(idx_i, i1)

        return p1, i1


class EqvarLayer(nn.Module):

    def __init__(self, n_nodes: list[int]):
        super().__init__()
        self.pp_layer = PPLayer(n_nodes[0], n_nodes[-1], activation=None)
        self.pi_layer = PIXLayer(n_nodes[0], n_nodes[-1], activation=None)
        self.ii_layer = IILayer(n_nodes[0], n_nodes[-1], activation=None)
        self.ip_layer = IPLayer()

        self.scale_layer = ScaleLayer()

    def forward(self, idx_i, idx_j, px, diff):

        px = self.pp_layer(px)
        ix = self.pi_layer(px, idx_i, idx_j)
        ix = self.scale_layer(ix, diff)
        ix = self.ii_layer(ix)
        px = self.ip_layer(idx_i, ix)

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
        rank: int,
        pp_nodes: list[int],
        pi_nodes: list[int],
        ii_nodes: list[int],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.rank = rank
        assert rank in [1, 3, 5], NotImplementedError("Only rank 1, 3, 5 are supported")
        self.p1_layer = InvarLayer(pp_nodes, pi_nodes, ii_nodes, activation)
        if rank >= 3:
            self.p3_layer = EqvarLayer([pp_nodes[0], pp_nodes[-1]])
        if rank >= 5:
            self.p5_layer = EqvarLayer([pp_nodes[0], pp_nodes[-1]])

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

        if self.rank >= 3:
            p3 = inputs["pinet", "p1"]
            diff_p3 = inputs[alias.pair_diff]
            p3, i3 = self.p3_layer(pair_i, pair_j, p3, basis, diff_p3)
            inputs["pinet", "p3"] = p3
            inputs["pinet", "i3"] = i3
        if self.rank >= 5:
            p5 = inputs["pinet", "p5"]
            diff_p5 = inputs[diff_p5]
            p5, i5 = self.p5_layer(pair_i, pair_j, p5, basis, diff_p5)
            inputs["pinet", "p5"] = p5
            inputs["pinet", "i5"] = i5
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
        max_atomtypes: int = 100,
        rank: int = 1,
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
        self.labels.set(
            "p5",
            torch.Tensor,
            "unit",
            "tensorial property",
            ("n_atoms", "n_components", "n_channels"),
        )

        self.rank = rank
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
                GCBlock(rank, pp_nodes, pi_nodes, ii_nodes, activation)
                for _ in range(depth)
            ]
        )

        self.res_update = ResUpdate()

    def forward(self, inputs: dict[str, torch.Tensor]) -> None:

        # get tensors from input dictionary
        atomic_numbers = inputs[alias.Z]
        n_atoms = atomic_numbers.shape[0]
        xyz = inputs[alias.xyz]
        idx_i = inputs[alias.pair_i]
        idx_j = inputs[alias.pair_j]
        offsets = inputs[alias.pair_offset]
        r_ij = xyz[idx_j] - xyz[idx_i] + offsets
        d_ij = torch.norm(r_ij, dim=-1)

        basis = self.basis_fn(d_ij)
        fc = self.cutoff_fn(d_ij)

        inputs["pinet", "basis"] = basis * fc[..., None]
        inputs["pinet", "p1"] = self.embedding(atomic_numbers)[:, None, :]
        n_channels = inputs["pinet", "p1"].shape[-1]

        inputs["pinet", "p1"] = self.before_gc_block_layer(inputs["pinet", "p1"])

        if self.rank >= 3:
            inputs["pinet", "p3"] = torch.zeros([n_atoms, 3, n_channels])
        if self.rank >= 5:
            inputs["pinet", "p5"] = torch.zeros([n_atoms, 5, n_channels])

        x = r_ij[:, 0]
        y = r_ij[:, 1]
        z = r_ij[:, 2]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        inputs["pinet", "diff_p5"] = torch.stack(
            [
                2 / 3 * x2 - 1 / 3 * (y2 + z2),
                2 / 3 * y2 - 1 / 3 * (x2 + z2),
                x * y,
                x * z,
                y * z,
            ],
            axis=1,
        )

        for i in range(self.depth):
            inputs = self.gc_blocks[i](inputs)

            inputs["pinet", "p1"] = self.res_update(inputs["pinet", "p1"], inputs["pinet", "p1"])
            if self.rank >= 3:
                inputs["pinet", "p3"] = self.res_update(inputs["pinet", "p3"], inputs["pinet", "p3"])
            if self.rank >= 5:
                inputs["pinet", "p5"] = self.res_update(inputs["pinet", "p5"], inputs["p5"])

        return inputs
