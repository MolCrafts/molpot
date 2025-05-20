from molpot.potential.base import Potential
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from molpot import alias
from .base import Dense, FeedForward


class PILayer(nn.Module):

    def __init__(self, *n_nodes: int, bias=True, activation: Callable | None = F.tanh):
        super().__init__()
        n_nodes = [n_nodes[0] * 2] + list(n_nodes[1:])
        self.mlp = FeedForward(*n_nodes, activation=activation, bias=bias)

    def forward(self, p1, pair_i, pair_j, basis):

        p1_i = p1[pair_i]
        p1_j = p1[pair_j]

        # inter = p1_i + p1_j
        inter = torch.concat([p1_i, p1_j], dim=-1)
        inter = self.mlp(inter)
        inter = torch.einsum(
            "icb, ib->ic", inter.reshape(-1, p1_i.shape[-1], basis.shape[-1]), basis
        )  # icb := (n_pairs, n_channels, n_basis)
        return inter[:, None, :]  # (n_pairs, 1, n_channels)


class IPLayer(nn.Module):

    def forward(self, ix, pair_i, px):
        n_atoms = px.shape[0]
        return torch.index_add(
            torch.zeros((n_atoms, *ix.shape[1:]), dtype=px.dtype, device=px.device),
            0,
            pair_i,
            ix,
        )


class PIXLayer(nn.Module):

    def __init__(
        self,
        *n_nodes: int,
    ):
        super().__init__()
        self.w = nn.Linear(*n_nodes, bias=False)

    def forward(self, px, pair_i, pair_j):

        px_j = px[pair_j]
        return px_j


class ScaleLayer(nn.Module):

    def forward(self, px, p1):
        return px * p1  # 'pcf,pcf->pcf'


class DotLayer(nn.Module):

    def forward(self, p):
        return torch.einsum("ixr, ixr->ir", p, p)[:, None, :]


class InvarLayer(nn.Module):

    def __init__(
        self,
        pp_nodes: list[int],
        pi_nodes: list[int],
        ii_nodes: list[int],
        activation: Callable = F.tanh,
    ):
        super().__init__()
        self.pi_layer = PILayer(*pi_nodes, bias=True, activation=activation)
        self.ii_layer = FeedForward(*ii_nodes, activation=activation, bias=False)
        self.ip_layer = IPLayer()
        self.pp_layer = FeedForward(*pp_nodes, bias=True, activation=activation)

    def forward(self, p1, pair_i, pair_j, basis):
        i1 = self.pi_layer(p1, pair_i, pair_j, basis)
        i1 = self.ii_layer(i1)
        p1 = self.ip_layer(i1, pair_i, p1)
        p1 = self.pp_layer(p1)

        return p1, i1


class EqvarLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.pp_layer = nn.Linear(in_features, out_features, bias=False)
        self.pi_layer = PIXLayer(in_features, out_features)
        self.ii_layer = FeedForward(
            in_features, out_features, activation=None, bias=False
        )
        self.ip_layer = IPLayer()

        self.scale_layer = ScaleLayer()

    def forward(self, px, pair_i, pair_j, diff, i1):

        ix = self.pi_layer(px, pair_i, pair_j)
        # iilayer
        ix = self.scale_layer(ix, i1)
        scaled_diff = self.scale_layer(diff[..., None], i1)
        ix = ix + scaled_diff
        px = self.ip_layer(ix, pair_i, px)
        px = self.pp_layer(px)

        return px, ix


class GCBlock1(nn.Module):

    def __init__(
        self,
        pp_nodes: list[int],
        pi_nodes: list[int],
        ii_nodes: list[int],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()
        self.p1_layer = InvarLayer(pp_nodes, pi_nodes, ii_nodes, activation)

    def forward(self, p1, pair_i, pair_j, basis, *args) -> dict[str, torch.Tensor]:
        p1, i1 = self.p1_layer(p1, pair_i, pair_j, basis)

        return (p1,), (i1,)


class GCBlock3(nn.Module):

    def __init__(
        self,
        pp_nodes: list[int],
        pi_nodes: list[int],
        ii_nodes: list[int],
        activation: Callable | None = F.tanh,
    ):
        super().__init__()

        ii1_nodes = ii_nodes.copy()
        pp1_nodes = pp_nodes.copy()
        ii1_nodes[-1] *= 2
        pp1_nodes[0] *= 2
        self.p1_layer = InvarLayer(pp1_nodes, pi_nodes, ii1_nodes, activation)
        self.p3_layer = EqvarLayer(pp_nodes[0], pp_nodes[-1])
        self.n_features = pp_nodes[-1]
        # n_props = 2
        self.pp_layer = nn.Linear(self.n_features * 2, self.n_features * 2)
        self.scale_layer = ScaleLayer()
        self.dot_layer = DotLayer()

    def forward(self, p1, p3, pair_i, pair_j, basis, diff):

        p1, i1 = self.p1_layer(p1, pair_i, pair_j, basis)
        i1s = torch.split(i1, int(i1.shape[-1] / 2), dim=-1)
        p3, i3 = self.p3_layer(p3, pair_i, pair_j, diff, i1s[1])

        px = self.pp_layer(torch.concat([p1, self.dot_layer(p3)], dim=-1))
        p1t1, p3_scale = torch.split(
            px,
            self.n_features,
            dim=-1,
        )
        p3t1 = self.scale_layer(p3, p3_scale)

        return (p1t1, p3t1), (i1s[0], i3)


# class OutLayer(nn.Module):

#     def __init__(self, out_nodes: list[int], out_units: int):
#         super().__init__()
#         if len(out_nodes) < 2:
#             raise ValueError("out_nodes must have at least 2 elements")
#         elif len(out_nodes) == 2:
#             out_nodes.append(out_nodes[-1])
#         self.ff_layer = FeedForward(
#             *out_nodes[:-1], activation=None, bias=True, last_bias=True
#         )
#         self.out_units = Dense(out_nodes[-1], out_units, activation=None, bias=False)

#     def forward(self, px, prev_px):
#         px = self.ff_layer(px)
#         return self.out_units(px) + prev_px


class ResUpdate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, old, new):
        return old + new


class PiNet2(Potential):
    """
    PiNet2: A neural network-based potential model for modeling interactions in molecular systems.
    Attributes:
        in_keys (list): Input keys required for the model, including atomic numbers, pairwise differences, and indices.
        out_keys (list): Output keys produced by the model, including processed features and interaction terms.
    Args:
        depth (int): Number of graph convolution layers in the model.
        basis_fn (Callable | None): Basis function used for feature transformation.
        cutoff_fn (Callable | None): Cutoff function applied to pairwise distances.
        pp_nodes (list[int]): Number of nodes in pair-pair interaction layers.
        pi_nodes (list[int]): Number of nodes in pair-atom interaction layers.
        ii_nodes (list[int]): Number of nodes in atom-atom interaction layers.
        activation (Callable | None): Activation function used in the network (default: `torch.tanh`).
        max_atomtypes (int): Maximum number of atom types supported by the embedding layer.
    Methods:
        forward(Z, pair_diff, pair_i, pair_j):
            Performs a forward pass through the PiNet2 model.
            Args:
                Z (torch.Tensor): Atomic numbers of the atoms in the system.
                pair_diff (torch.Tensor): Pairwise differences between atomic positions.
                pair_i (torch.Tensor): Indices of the first atoms in the pairs.
                pair_j (torch.Tensor): Indices of the second atoms in the pairs.
            Returns:
                tuple: Processed features and interaction terms (p1, p3t1, i1, i3).
        cite():
            Returns the citation for the PiNet2 model.
            Returns:
                str: Citation string for the PiNet2 model.
    """

    in_keys = [
        alias.Z,
        alias.pair_diff,
        alias.pair_i,
        alias.pair_j,
    ]
    out_keys = [("pinet", "p1"), ("pinet", "p3"), ("pinet", "i1"), ("pinet", "i3")]

    def __init__(
        self,
        depth: int,
        basis_fn: Callable | None = None,
        cutoff_fn: Callable | None = None,
        pp_nodes: list[int] = [16, 16],
        pi_nodes: list[int] = [16, 16],
        ii_nodes: list[int] = [16, 16],
        activation: Callable | None = F.tanh,
        max_atomtypes: int = 100,
    ):
        super().__init__()

        self.register_buffer("depth", torch.tensor(depth))
        self.basis_fn = basis_fn
        self.cutoff_fn = cutoff_fn
        self.n_basis = self.basis_fn.n_rbf

        self.embedding = nn.Embedding(max_atomtypes, self.n_basis, padding_idx=0)

        pi_nodes[-1] *= self.n_basis
        self.before_gc_block_layer = nn.Linear(self.n_basis, pp_nodes[0])

        self.gc_blocks = nn.ModuleList(
            [GCBlock3(pp_nodes, pi_nodes, ii_nodes, activation) for _ in range(depth)]
        )
        # self.out_layers = nn.ModuleList([OutLayer(out_nodes, out_units) for _ in range(depth)])

        self.res_update = ResUpdate()

    def forward(self, Z, pair_diff, pair_i, pair_j) -> None:
        n_atoms = Z.shape[0]
        pair_diff.requires_grad_()
        pair_dist = torch.linalg.norm(pair_diff, dim=-1)  # (n_pairs, )
        pair_i = pair_i.to(torch.int64)  # for scatter
        pair_j = pair_j.to(torch.int64)
        norm_pair_diff = pair_diff / pair_dist[:, None]

        basis = self.basis_fn(pair_dist)
        fc = self.cutoff_fn(pair_dist)

        basis = basis * fc[..., None]
        p1 = self.embedding(Z)[:, None, :]

        # (n_atoms, 1, ...) -> (n_atoms, 1, n_basis)
        p1 = self.before_gc_block_layer(p1)
        p3 = torch.zeros([n_atoms, 3, p1.shape[-1]], dtype=p1.dtype, device=p1.device)

        for i in range(self.depth.item()):
            (p1t1, p3t1), (i1, i3) = self.gc_blocks[i](
                p1, p3, pair_i, pair_j, basis, norm_pair_diff
            )
            p1 = self.res_update(p1, p1t1)
            p3 = self.res_update(p3, p3t1)
        return (p1, p3t1, i1, i3)
    
    def cite(self):
        return "Li, J.; Knijff, L.; Zhang, Z.-Y.; Andersson, L.; Zhang, C. PiNN: Equivariant Neural Network Suite for Modelling Electrochemical Systems. J. Chem. Theory Comput., 2025, 21: 1382."


class PiNet1(Potential):

    in_keys = [
        alias.Z,
        alias.pair_diff,
        alias.pair_i,
        alias.pair_j,
    ]
    out_keys = [("pinet", "p1"), ("pinet", "i1")]

    def __init__(
        self,
        depth: int,
        basis_fn: Callable | None = None,
        cutoff_fn: Callable | None = None,
        pp_nodes: list[int] = [16, 16],
        pi_nodes: list[int] = [16, 16],
        ii_nodes: list[int] = [16, 16],
        out_nodes: list[int] = [16, 16],
        out_units: int = 1,
        activation: Callable | None = F.tanh,
        max_atomtypes: int = 100,
    ):
        super().__init__()

        self.register_buffer("depth", torch.tensor(depth))
        self.basis_fn = basis_fn
        self.cutoff_fn = cutoff_fn
        self.n_basis = self.basis_fn.n_rbf

        self.embedding = nn.Embedding(max_atomtypes, self.n_basis, padding_idx=0)

        pi_nodes[-1] *= self.n_basis
        self.before_gc_block_layer = Dense(
            self.n_basis, pp_nodes[0], activation=activation
        )

        self.gc_blocks = nn.ModuleList(
            [
                GCBlock1(pp_nodes, pi_nodes, ii_nodes, activation=activation)
                for _ in range(depth)
            ]
        )
        # self.out_layers = nn.ModuleList([OutLayer(out_nodes, out_units) for _ in range(depth)])

        self.res_update = ResUpdate()

    def forward(self, Z, pair_diff, pair_i, pair_j) -> None:
        # n_atoms = Z.shape[0]
        # pair_diff.requires_grad_()
        pair_dist = torch.linalg.norm(pair_diff, dim=-1)  # (n_pairs, )
        pair_i = pair_i.to(torch.int64)  # for scatter
        pair_j = pair_j.to(torch.int64)
        norm_pair_diff = pair_diff / pair_dist[:, None]

        basis = self.basis_fn(pair_dist)
        fc = self.cutoff_fn(pair_dist)

        basis = basis * fc[..., None]
        p1 = self.embedding(Z)[:, None, :]
        # (n_atoms, 1, n_basis) -> (n_atoms, 1, pp_nodes[0])
        p1 = self.before_gc_block_layer(p1)

        # output = 0.0
        for i in range(self.depth.item()):
            (p1t1,), (i1,) = self.gc_blocks[i](
                p1, pair_i, pair_j, basis, norm_pair_diff
            )
            # output = self.out_layers[i](p1t1, output)
            p1 = self.res_update(p1, p1t1)
        return (p1, i1)

    def cite(self):
        return "Shao, Y.; Hellström, M.; Mitev, P. D.; Knijff, L.; Zhang, C. PiNN: A Python Library for Building Atomic Neural Networks of Molecules and Materials. J. Chem. Inf. Model., 2020, 60: 1184."