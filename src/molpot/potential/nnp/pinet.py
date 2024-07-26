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

        inter = prop_i + basis + prop_j
        inter = self.mlp(inter)
        return inter


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
        ii = self.ii_layer(i1)
        p1 = self.ip_layer(idx_i, ii)

        return p1, i1


class EqvarLayer(nn.Module):

    def __init__(self, n_nodes: list[int]):
        super().__init__()
        self.pp_layer = PPLayer(n_nodes[0], n_nodes[-1], activation=None)
        self.pi_layer = PILayer(n_nodes[0], n_nodes[-1], activation=None)
        self.ii_layer = IILayer(n_nodes[0], n_nodes[-1], activation=None)
        self.ip_layer = IPLayer()

        self.scale_layer = ScaleLayer()

    def forward(self, idx_i, idx_j, px, basis, diff):

        px = self.pp_layer(px)
        ix = self.pi_layer(px, idx_i, idx_j, basis)
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

    def forward(self, inputs, outputs) -> dict[str, torch.Tensor]:
        pair_i = inputs["pair_i"]
        pair_j = inputs["pair_j"]
        basis = inputs["basis"]
        p1 = inputs["p1"]
        p1, i1 = self.p1_layer(pair_i, pair_j, p1, basis)
        outputs["p1"] = p1
        outputs["i1"] = i1

        if self.rank >= 3:
            p3 = outputs["p3"]
            diff_p3 = inputs["norm_diff"]
            p3, i3 = self.p3_layer(pair_i, pair_j, p3, basis, diff_p3)
            outputs["p3"] = p3
            outputs["i3"] = i3
        if self.rank >= 5:
            p5 = outputs["p5"]
            diff_p5 = inputs["diff_p5"]
            p5, i5 = self.p5_layer(pair_i, pair_j, p5, basis, diff_p5)
            outputs["p5"] = p5
            outputs["i5"] = i5
        return inputs, outputs


class ResUpdate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, old, new):
        return old + new


class PiNet(nn.Module):

    def __init__(
        self,
        n_atom_basis: int,
        depth: int,
        basis_fn: Callable | None = None,
        cutoff_fn: Callable | None = None,
        pp_nodes: int = [16, 16],
        pi_nodes: int = [16, 16],
        ii_nodes: int = [16, 16],
        activation: Callable | None = F.tanh,
        max_z: int = 101,
        rank: int = 1,
    ):
        super().__init__()

        self.labels = NameSpace("pinet")
        self.labels.set(
            "p1", torch.Tensor, "unit", "scalar property", (None, 1, n_atom_basis)
        )
        self.labels.set(
            "p3", torch.Tensor, "unit", "vectorial property", (None, 3, n_atom_basis)
        )
        self.labels.set(
            "p5", torch.Tensor, "unit", "tensorial property", (None, 5, n_atom_basis)
        )

        self.rank = rank
        self.n_atom_basis = n_atom_basis

        self.depth = depth
        self.basis_fn = basis_fn
        self.cutoff_fn = cutoff_fn

        assert basis_fn.n_rbf == pp_nodes[0]

        pp_nodes = [n_atom_basis] + pp_nodes
        ii_nodes = ii_nodes + [n_atom_basis]

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        # self.basis_transformation = nn.Linear(basis_fn.n_rbf, n_atom_basis)

        self.gc_blocks = nn.ModuleList(
            [
                GCBlock(rank, pp_nodes, pi_nodes, ii_nodes, activation)
                for _ in range(depth)
            ]
        )

        self.res_update = ResUpdate()

    def forward(
        self, inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> None:

        # get tensors from input dictionary
        atomic_numbers = inputs[alias.Z]
        n_atoms = atomic_numbers.shape[0]
        xyz = inputs[alias.xyz]
        idx_i = inputs[alias.pair_i]
        idx_j = inputs[alias.pair_j]
        offsets = inputs[alias.offsets]
        r_ij = xyz[idx_j] - xyz[idx_i] + offsets
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)

        basis = self.basis_fn(d_ij)
        fc = self.cutoff_fn(d_ij)
        # inputs["basis"] = self.basis_transformation(basis * fc[..., None])
        inputs["basis"] = basis * fc[..., None]
        inputs["p1"] = self.embedding(atomic_numbers)[:, None, :]
        if self.rank >= 3:
            inputs["p3"] = torch.zeros([n_atoms, 3, self.n_atom_basis])
        if self.rank >= 5:
            inputs["p5"] = torch.zeros([n_atoms, 5, self.n_atom_basis])

        inputs["diff_p5"] = d_ij
        x = r_ij[:, 0]
        y = r_ij[:, 1]
        z = r_ij[:, 2]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        inputs["diff_p5"] = torch.stack(
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
            inputs, outputs = self.gc_blocks[i](inputs, outputs)
            outputs["p1"] = self.res_update(outputs["p1"], inputs["p1"])
            if self.rank >= 3:
                outputs["p3"] = self.res_update(outputs["p3"], inputs["p3"])
            if self.rank >= 5:
                outputs["p5"] = self.res_update(outputs["p5"], inputs["p5"])

        return inputs, outputs
