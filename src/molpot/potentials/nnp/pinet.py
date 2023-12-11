# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-12
# version: 0.0.1

from typing import Callable
from ..base import Potential
import torch
from torch import nn
from .layers import PolynomialBasis, GaussianBasis, CutoffFunc
from .layers import ANNOutput
import molpot as mpot

__all__ = [
    "PiNet",
    "FFLayer",
    "PILayer",
    "IPLayer",
    "OutLayer",
    "GCBlock",
    "ResUpdate",
    "PIXLayer",
    "DotLayer",
    "ScaleLayer",
]


class FFLayer(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_features: list = [64, 64],
        activation: nn.Module = nn.ReLU(),
        bias=True,
    ):
        in_features = [in_feature] + out_features[:-1]
        super().__init__()
        modules = []
        for i, o in zip(in_features, out_features):
            modules.append(nn.Linear(i, o, bias=bias))
            modules.append(activation)
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.dense_layers(x)


class PILayer(nn.Module):
    def __init__(self, in_feature: int, out_features=[64], n_basis=4):
        super(PILayer, self).__init__()

        self.out_features = out_features
        out_nodes = out_features.copy()
        out_nodes[-1] *= n_basis
        self.n_basis = n_basis

        self.ff_layer = FFLayer(in_feature*2, out_nodes, activation=nn.Identity())

    def forward(self, prop, idx_i, idx_j, basis):
        prop_i = prop[idx_i]
        prop_j = prop[idx_j]

        inter = torch.cat([prop_i, prop_j], axis=-1)
        inter = self.ff_layer(inter)
        inter = inter.reshape((-1, self.out_features[-1], self.n_basis))
        inter = torch.einsum("pcb,pb->pc", inter, basis)
        return inter


class PIXLayer(nn.Module):
    def __init__(self):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super(PIXLayer, self).__init__()

    def forword(self, px, ind_2):
        """
        PILayer take a list of three tensors as input:

        - ind_2: [sparse indices](layers.md#sparse-indices) of pairs with shape `(n_pairs, 2)`
        - prop: property tensor with shape `(n_atoms, n_prop)`

        Args:
            tensors (list of tensors): list of `[ind_2, prop, basis]` tensors

        Returns:
            inter (tensor): interaction tensor with shape `(n_pairs, n_nodes[-1])`
        """
        # ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        # px_i = px[ind_i]
        px_j = px[ind_j]

        return px_j


class DotLayer(nn.Module):
    def __init__(self):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super(DotLayer, self).__init__()

    def forword(self, tensor):
        return torch.einsum("ixr,ixr->ir", tensor, tensor)


class ScaleLayer(nn.Module):
    def __init__(self):
        super(ScaleLayer, self).__init__()

    def forward(self, tensor):
        px, p1 = tensor
        return px * p1[:, None, :]


class IPLayer(nn.Module):
    def __init__(self):
        super(IPLayer, self).__init__()

    def forward(self, ind_2, prop, inter):
        return torch.scatter_add(
            torch.zeros(prop.shape[0], prop.shape[-1]), 0, ind_2[:, 0], inter
        )


class OutLayer(nn.Module):
    def __init__(self, in_features: int, out_features):
        super(OutLayer, self).__init__()
        self.ff_layer = FFLayer(in_features, out_features[:-1])
        self.out_layer = nn.Linear(out_features[-2], out_features[-1], bias=False)

    def forward(self, p1, prev_output):
        p1 = self.ff_layer(p1)
        output = self.out_layer(p1) + prev_output
        return output


class GCBlock(nn.Module):
    def __init__(self, pp_nodes, pi_nodes, ii_nodes, activation=nn.ReLU()):
        super().__init__()
        ii_nodes = ii_nodes.copy()
        ii_nodes[-1] *= 3
        self.pp1_layer = FFLayer(1, pp_nodes, activation)
        self.pi1_layer = PILayer(1, pi_nodes)
        self.ii1_layer = FFLayer(1, ii_nodes, activation=activation, bias=False)
        self.ip1_layer = IPLayer()

        self.pix_layer = PIXLayer()
        self.ii3_layer = FFLayer(3, ii_nodes, activation=activation, bias=False)
        self.ip3_layer = IPLayer()

        self.dot_layer = DotLayer()

        self.scale1_layer = ScaleLayer()
        self.scale2_layer = ScaleLayer()
        self.scale3_layer = ScaleLayer()

    def forword(self, p1, p3, basis, diff, ind_2):
        p1 = self.pp1_layer(p1)
        i1 = self.pi1_layer(ind_2, p1, basis)
        i1 = self.ii1_layer(i1)
        i1_1, i1_2, i1_3 = torch.split(i1, 3, axis=-1)
        p1 = self.ip1_layer([ind_2, p1, i1_2])

        i3 = self.pix_layer([ind_2, p3])
        i3 = self.scale1_layer([i3, i1_3])
        scaled_diff = self.scale2_layer([diff[:, :, None], i1_1])
        i3 = i3 + scaled_diff
        p3 = self.ip3_layer([ind_2, p3, i3])

        p1t1 = self.dot_layer(p3) + p1
        p3t1 = self.scale3_layer([p3, p1t1])

        return p1t1, p3t1


class ResUpdate(nn.Module):
    def __init__(self):
        super(ResUpdate, self).__init__()

    def forward(self, tensors):
        old, new = tensors
        return old + new


class PiNet(Potential):
    """This class implements the Keras Model for the PiNet network."""

    def __init__(
        self,
        rc=5.0,
        cutoff_type="f1",
        basis_type="polynomial",
        n_basis=4,
        gamma=3.0,
        center=None,
        pp_nodes=[16, 16],
        pi_nodes=[16, 16],
        ii_nodes=[16, 16],
        out_nodes=[16, 16],
        out_pool=False,
        act="tanh",
        depth=4,
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
        super(PiNet, self).__init__('PiNet')
        if act == "tanh":
            act = nn.Tanh()
        self.depth = depth
        self.alias = mpot.alias("PiNet")
        self.alias.set("p1", "_pinet_p1", torch.Tensor, None, "invariant property")
        self.alias.set("p3", "_pinet_p3", torch.Tensor, None, "equalvariant property")
        self.cutoff = CutoffFunc(rc, cutoff_type)
        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        self.res_update = nn.Sequential(*[ResUpdate() for _ in range(depth)])
        gc_blocks = [GCBlock([], pi_nodes, ii_nodes, activation=act)]
        gc_blocks += [
            GCBlock(pp_nodes, pi_nodes, ii_nodes, activation=act)
            for _ in range(depth - 1)
        ]
        self.gc_blocks = nn.Sequential(*gc_blocks)
        self.out_layers = [OutLayer(1, out_nodes) for _ in range(depth)]
        self.ann_output = ANNOutput(out_pool)

    def forward(self, tensors):
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
        tensors["p3"] = torch.zeros([tensors["ind_1"].shape[0], 3, 1])
        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        output = 0.0
        for i in range(self.depth):
            p1, p3 = self.gc_blocks[i](
                {
                    self.alias.ind_2: tensors["ind_2"],
                    self.alias.p1: tensors["p1"],
                    self.alias.p3: tensors["p3"],
                    self.alias.diff: tensors["diff"],
                    self.alias.basis: basis,
                }
            )
            output = self.out_layers[i](p1, output)
            tensors["p1"] = self.res_update[i](tensors["p1"], p1)
            # tensors["p3"] = self.res_update[i]([tensors["p3"], p3])

        output = self.ann_output([tensors["ind_1"], output])
        return output
