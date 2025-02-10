import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Literal
from molpot import NameSpace, alias
from .base import FeedForward


class PILayer(nn.Module):

    def __init__(self, *n_nodes: int, activation: Callable | None = F.tanh):
        super().__init__()
        self.mlp = FeedForward(*n_nodes, activation=activation, bias=False)

    def forward(self, p1, pair_i, pair_j, basis):

        p1_i = p1[pair_i]
        p1_j = p1[pair_j]

        inter = p1_i + p1_j
        inter = self.mlp(inter)
        inter = torch.einsum(
            "icb, ib->ic", inter.reshape(-1, p1_i.shape[-1], basis.shape[-1]), basis
        )  # icb := (n_pairs, n_channels, n_basis)
        return inter[:, None, :]  # (n_pairs, 1, n_channels)


class IPLayer(nn.Module):

    def forward(self, i1, pair_i, p1):
        return torch.index_add(
            torch.zeros_like(p1, dtype=p1.dtype, device=p1.device), 0, pair_i, i1
        )


class PIXLayer(nn.Module):

    def __init__(
        self,
        *n_nodes: int,
    ):
        super().__init__()
        self.w = nn.Linear(*n_nodes, bias=False)

    def forward(self, px, pair_i, pair_j):

        px_i = px[pair_i]
        px_j = px[pair_j]
        return self.w(px_i + px_j)

class ScaleLayer(nn.Module):

    def forward(self, px, p1):
        return px * p1  # 'pcf,pcf->pcf'


class SelfDotLayer(nn.Module):

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
        self.pp_layer = FeedForward(*pp_nodes, activation=activation, bias=True)
        self.pi_layer = PILayer(*pi_nodes, activation=activation)
        self.ii_layer = FeedForward(*ii_nodes, activation=activation, bias=False)
        self.ip_layer = IPLayer()

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
        self.p1_layer = InvarLayer(pp_nodes, pi_nodes, ii_nodes, activation)
        self.p3_layer = EqvarLayer(pp_nodes[0], pp_nodes[-1])
        self.n_features = pp_nodes[-1]
        # n_props = 2
        self.pp_layer = nn.Linear(self.n_features, self.n_features)
        self.scale_layer = ScaleLayer()
        self.dot_layer = SelfDotLayer()

    def forward(self, p1, p3, pair_i, pair_j, basis, diff):

        p1, i1 = self.p1_layer(p1, pair_i, pair_j, basis)
        p3, i3 = self.p3_layer(p3, pair_i, pair_j, diff, i1)

        px = self.pp_layer(torch.concat([p1, self.dot_layer(p3)], dim=-1))
        p1t1, p3_scale = torch.split(px, [1, 1], dim=-1)
        p3t1 = self.scale_layer(p3, p3_scale)

        return (p1t1, p3t1), (i1, i3)


class OutLayer(nn.Module):

    def __init__(self, out_nodes: list[int]):
        super().__init__()
        if len(out_nodes) < 2:
            raise ValueError("out_nodes must have at least 2 elements")
        elif len(out_nodes) == 2:
            out_nodes.append(out_nodes[-1])
        self.ff_layer = FeedForward(*out_nodes[:-1])
        # self.out_units = nn.Linear(out_nodes[-2], out_nodes[-1])

    def forward(self, px, prev_px):
        px = self.ff_layer(px)
        # px = self.out_units(px)
        return px + prev_px


class ResUpdate(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, old, new):
        return old + new


class PiNet2(nn.Module):

    in_keys = [
        alias.Z,
        alias.pair_diff,
        alias.pair_i,
        alias.pair_j,
    ]
    out_keys = [("pinet2", "p1"), ("pinet2", "i1"), ("pinet2", "p3"), ("pinet2", "i3")]

    def __init__(
        self,
        depth: int,
        basis_fn: Callable | None = None,
        cutoff_fn: Callable | None = None,
        pp_nodes: list[int] = [16, 16],
        pi_nodes: list[int] = [16, 16],
        ii_nodes: list[int] = [16, 16],
        out_nodes: list[int] = [16, 16],
        activation: Callable | None = F.tanh,
        max_atomtypes: int = 100,
    ):
        super().__init__()
        self.labels = NameSpace("pinet")
        self.labels.set(
            "p1",
            "scalar property",
            float,
            shape=(None, 1, ii_nodes[-1]),
        )
        self.labels.set(
            "p3",
            "vectorial property",
            float,
            shape=(None, 3, ii_nodes[-1]),
        )

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
        self.out_layers = nn.ModuleList([OutLayer(out_nodes) for _ in range(depth)])

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
        i1 = torch.zeros(
            [pair_i.shape[0], 1, p1.shape[-1]], dtype=p1.dtype, device=p1.device
        )
        i3 = torch.zeros(
            [pair_i.shape[0], 3, p1.shape[-1]], dtype=p1.dtype, device=p1.device
        )

        output = 0.0
        for i in range(self.depth.item()):
            (p1t1, p3t1), (i1, i3) = self.gc_blocks[i](
                p1, p3, pair_i, pair_j, basis, norm_pair_diff
            )
            output = self.out_layers[i](p1t1, output)
            p1 = self.res_update(p1, p1t1)
            p3 = self.res_update(p3, p3t1)
        return (output, p3t1, i1, i3)


class PiNet1(nn.Module): ...
