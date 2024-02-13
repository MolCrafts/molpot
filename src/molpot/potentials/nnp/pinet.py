# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-12
# version: 0.0.1

from .readout import Atomwise
from ..base import NNPotential
import torch
from torch import nn
from .layers import Dense, build_mlp
import molpot as mpot
from molpot import alias
from typing import Optional, Callable, Sequence
import torch.nn.functional as F


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
        prop_j = prop[idx_j]

        inter = torch.cat([prop_i, prop_j], axis=-1)
        inter = self.ff_layer(inter)  # NOTE: weight?
        inter = inter.reshape([*inter.shape[:-1], self.n_neurons[-2], self.n_basis])
        inter = torch.einsum("i...c,ic->i...", inter, basis)
        return inter


class DotLayer(nn.Module):
    def __init__(self):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super().__init__()

    def forward(self, px):
        return torch.einsum("i...c,i...c->ic", px, px)


class ScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, px, p1):
        return torch.einsum("i...c,i...c->i...c", px, p1)


class IPLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i, idx_i, p):
        
        p.index_add(
            0, idx_i, i
        )
        return p


class OutLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_hidden, n_layers):
        super().__init__()
        self.ff_layer = build_mlp(in_features, out_features, n_hidden, n_layers)

    def forward(self, prev_output, p1, p3):
        p1 = self.ff_layer(p1)
        output = p1 + prev_output
        return output


# class ResUpdate(nn.Module):
#     def __init__(self, in_features: int, out_features: int, n_hidden, n_layers):
#         super().__init__()
#         self.transform = Dense(
#             in_features, out_features, n_hidden, n_layers, activation=None
#         )

#     def forward(self, old, new):
#         return self.transform(old) + new


class GCBlockP1(nn.Module):

    def __init__(self, pp_nodes, pi_nodes, ii_nodes, n_basis):
        super().__init__()
        self.pp_layer = build_mlp(pp_nodes)
        self.pi_layer = PILayer(pp_nodes[-1], pi_nodes, n_basis)
        self.ii_layer = build_mlp(ii_nodes)
        self.ip_layer = IPLayer()

    def forward(self, p, idx_i, idx_j, basis):
        p1 = self.pp_layer(p)
        i1 = self.pi_layer(p1, idx_i, idx_j, basis)
        i1 = self.ii_layer(i1)
        p  = self.ip_layer(i1, idx_i, p)
        return p


class GCBlock(nn.Module):
    def __init__(self, pp_nodes, pi_nodes, ii_nodes, n_basis, activation=nn.ReLU()):
        super().__init__()
        if pp_nodes:
            self.pp1_layer = build_mlp(
                pp_nodes[0],
                pp_nodes[-1],
                n_hidden=pp_nodes[1:-1],
                n_layers=len(pp_nodes),
                activation=activation,
            )
        else:
            self.pp1_layer = nn.Identity()
        self.pi1_layer = PILayer(
            pi_nodes[0], pi_nodes[-1], pi_nodes[1:-1], len(pi_nodes), n_basis
        )
        self.ii1_layer = build_mlp(
            ii_nodes[0],
            ii_nodes[-1],
            ii_nodes[1:-1],
            len(ii_nodes),
            activation=activation,
        )
        self.ip1_layer = IPLayer()

        if pp_nodes:
            self.pp3_layer = build_mlp(
                pp_nodes[0],
                pp_nodes[-1],
                n_hidden=pp_nodes[1:-1],
                n_layers=len(pp_nodes),
                activation=activation,
            )
        else:
            self.pp3_layer = nn.Identity()
        self.pi3_layer = PIXLayer()
        self.ii3_layer = build_mlp(
            ii_nodes[0],
            ii_nodes[-1],
            ii_nodes[1:-1],
            len(ii_nodes),
            activation=activation,
        )
        self.ip3_layer = IPLayer()

        self.pp5_layer = build_mlp(pp_nodes)
        self.pi5_layer = PIXLayer()
        self.ii5_layer = build_mlp(ii_nodes)
        self.ip5_layer = IPLayer()

        self.dot_layer = DotLayer()

        self.scale0_layer = ScaleLayer()
        self.scale1_layer = ScaleLayer()
        self.scale2_layer = ScaleLayer()
        self.scale3_layer = ScaleLayer()
        self.scale4_layer = ScaleLayer()

    def forward(self, p1, p3, p5, idx_i, idx_j, diff, basis):

        p1 = self.pp1_layer(p1)
        i1 = self.pi1_layer(p1, idx_i, idx_j, basis)
        i1 = self.ii1_layer(i1)
        i1_1, i1_2, i1_3, i1_4 = torch.split(i1, 4, dim=-1)
        p1 = self.ip1_layer(idx_i, p1, i1_1)
        scaled_diff = self.scale2_layer(diff)

        p3 = self.pp3_layer(p3)
        i3 = self.pi3_layer(p3, idx_i, idx_j)
        i3 = self.scale1_layer(i3)
        i3 = self.ii3_layer(i3)
        i3 = i3 + scaled_diff
        p3 = self.ip3_layer(idx_i, p3, i3)

        p5 = self.pp5_layer(p5)
        i5 = self.pi5_layer(p5, idx_i, idx_j)
        i5 = self.scale2_layer([i5, i1_4[:, None, :]])
        vvt = torch.einsum("ijl, ikl->ijkl", scaled_diff, scaled_diff)
        vvt_trace = torch.einsum("ijjl->il", vvt)
        S = vvt - torch.reshape(
            vvt_trace[:, :, None, None] * torch.eye(3).broadcast_to(vvt_trace) / 3,
            vvt.shape,
        )

        i5 = i5 + S
        p5 = self.ip5_layer(idx_i, p5, i5)

        next_p1 = p1 + self.dot_layer(p1)
        next_p3 = self.scale3_layer(p3, p1)

        return next_p1, next_p3


class PiNet(NNPotential):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        n_atom_basis: int,
        depth: int,
        radial_basis: nn.Module,
        cutoff_fn: Optional[Callable] = None,
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        out_pool=False,
        upto: int = 3,
        act="tanh",
    ):
        """
        Args:
            atom_types (list): elements for the one-hot embedding
            pp_nodes (list): number of nodes for PPLayer
            pi_nodes (list): number of nodes for PILayer
            ii_nodes (list): number of nodes for IILayer
            out_nodes (list): number of nodes for OutLayer
            out_pool (str): pool atomic outputs, see ANNOutput
            depth (int): number of interaction blocks
            rc (float): cutoff radius
            basis_type (string): basis function, can be "polynomial" or "gaussian"
            n_basis (int): number of basis functions to use
            gamma (float or array): width of gaussian function for gaussian basis
            center (float or array): center of gaussian function for gaussian basis
            cutoff_type (string): cutoff function to use with the basis.
            act (string): activation function to use
        """
        super().__init__("PiNet")
        if activation == "tanh":
            activation = nn.Tanh()
        self.depth = depth
        mpot.alias("pinet")
        alias.pinet.set("p1", "_pinet_p1", torch.Tensor, None, "invariant property")
        alias.pinet.set(
            "p3", "_pinet_p3", torch.Tensor, None, "rank1 equivalent property"
        )
        alias.pinet.set(
            "p5", "_pinet_p5", torch.Tensor, None, "rank2 equivalent property"
        )
        self.cutoff_fn = cutoff_fn
        self.radial_basis_fn = radial_basis
        self.n_basis = radial_basis.n_rbf
        self.n_atom_basis = n_atom_basis
        self.embbding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.res_update1 = nn.Sequential(*[ResUpdate() for _ in range(depth)])
        self.res_update3 = nn.Sequential(*[ResUpdate() for _ in range(depth)])
        gc_blocks = [
            GCBlock([], pi_nodes, ii_nodes, self.n_basis, activation=activation)
        ]
        gc_blocks += [
            GCBlock(pp_nodes, pi_nodes, ii_nodes, self.n_basis, activation=activation)
            for _ in range(depth - 1)
        ]
        self.gc_blocks = nn.Sequential(*gc_blocks)
        self.out_layers = [
            OutLayer(1, out_nodes[-1], out_nodes[:-1], len(out_nodes) + 1)
            for _ in range(depth)
        ]
        self.ann_output = Atomwise(out_pool)

    def forward(self, tensors: dict):
        """PiNet takes batches atomic data as input, the following keys are
        required in the input dictionary of tensors:

        - `ind_1`: [sparse indices](layers.md#sparse-indices) for the batched data, with shape `(n_atoms, 1)`;
        - `elems`: element (atomic numbers) for each atom, with shape `(n_atoms)`;
        - `coord`: coordintaes for each atom, with shape `(n_atoms, 3)`.

        Optionally, the input dataset can be processed with
        `PiNet.preprocess(tensors)`, which adds the following tensors to the
        dictionary:

        - `ind_2`: [sparse indices](layers.md#sparse-indices) for neighbour list, with shape `(n_pairs, 2)`;
        - `dist`: distances from the neighbour list, with shape `(n_pairs)`;
        - `diff`: distance vectors from the neighbour list, with shape `(n_pairs, 3)`;
        - `prop`: initial properties `(n_pairs, n_elems)`;

        Args:
            tensors (dict of tensors): input tensors

        Returns:
            output (tensor): output tensor with shape `[n_atoms, out_nodes]`
        """
        r_ij = tensors[alias.Rij]
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        fc = self.cutoff_fn(tensors[alias.Rij])
        basis = self.radial_basis_fn(dir_ij) * fc[..., None]
        for i in range(self.depth):
            p1, p3, p5 = self.gc_blocks[i](
                tensors[alias.pinet.p1],
                tensors[alias.pinet.p3],
                tensors[alias.pinet.p5],
                tensors[alias.idx_i],
                tensors[alias.idx_j],
                d_ij,
                basis,
            )
            tensors = self.out_layers[i](tensors[alias.idx_m], p1, p3, p5)
            tensors[alias.pinet.p1] = self.res_update1[i](tensors[alias.pinet.p1], p1)
            tensors[alias.pinet.p3] = self.res_update1[i](tensors[alias.pinet.p3], p3)
            tensors[alias.pinet.p5] = self.res_update1[i](tensors[alias.pinet.p5], p5)

        tensors[alias.pinet.p1] = self.ann_output(tensors)
        return tensors
