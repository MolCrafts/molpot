# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-12
# version: 0.0.1

from ..base import NNPotential
import torch
from torch import nn
from .layers import PolynomialBasis, GaussianBasis, CutoffFunc
from .layers import ANNOutput
import molpot as mpot

__all__ = [
    "PiNet",
]


class FFLayer(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_features: list = [64, 64],
        activation: nn.Module = nn.ReLU(),
        bias=True,
    ):
        super().__init__()
        in_features = [in_feature] + out_features[:-1]
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

        self.ff_layer = FFLayer(in_feature * 2, out_nodes, activation=nn.Identity(), bias=False)

    def forward(self, prop, idx_i, idx_j, basis):
        prop_i = prop[idx_i]
        prop_j = prop[idx_j]

        inter = torch.cat([prop_i, prop_j], axis=-1)
        inter = self.ff_layer(inter)
        inter = inter.reshape((-1, self.out_features[-1], self.n_basis))
        inter = torch.einsum("pcb,pb->pc", inter, basis)
        return inter


class PIXLayer(nn.Module):
    def __init__(self, weighted: bool, shape, **kwargs):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super(PIXLayer, self).__init__()
        self.weighted = weighted
        if weighted:
            self.wi = nn.Parameter(torch.zeros(shape[1][-1]))
            self.wj = nn.Parameter(torch.zeros(shape[1][-1]))

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
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        px_i = px[ind_i]
        px_j = px[ind_j]

        if self.weighted:
            return self.wi * px_i + self.wj * px_j
        else:
            return px_i - px_j


class DotLayer(nn.Module):
    def __init__(self, weighted: bool, shape, **kwargs):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super(DotLayer, self).__init__()
        self.weighted = weighted
        if weighted:
            self.wi = nn.Parameter(torch.zeros(shape[-1]))
            self.wj = nn.Parameter(torch.zeros(shape[-1]))

    def forword(self, tensor):
        if self.weighted:
            return torch.einsum("ixr,ixr->ir", self.wi * tensor, self.wj * tensor)


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


class PiNetGCBlock(nn.Module):
    def __init__(self, pp_nodes, pi_nodes, ii_nodes, activation=nn.ReLU()):
        super().__init__()
        self.pp_layer = FFLayer(pp_nodes[0], pp_nodes[1:])
        self.pi_layer = PILayer(pi_nodes[0], pi_nodes[1:])
        self.ii_layer = FFLayer(ii_nodes[0], ii_nodes[1:])
        self.ip_layer = IPLayer()

    def forword(self, prop, basis, diff, ind_2):
        prop = self.pp_layer(prop)
        inter = self.pi_layer(prop, ind_2, basis)
        inter = self.ii_layer(inter)
        prop = self.ip_layer(ind_2, prop, inter)
        return prop


class ResUpdate(nn.Module):
    def __init__(self):
        super(ResUpdate, self).__init__()

    def forward(self, tensors):
        old, new = tensors
        return old + new


class PiNet(NNPotential):
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
        super(PiNet, self).__init__("PiNet")
        if act == "tanh":
            act = nn.Tanh()
        self.depth = depth
        self.alias = mpot.alias("PiNet")
        self.alias.set("p1", "_pinet_p1", torch.Tensor, None, "invariant property")
        self.cutoff = CutoffFunc(rc, cutoff_type)
        if basis_type == "polynomial":
            self.basis_fn = PolynomialBasis(n_basis)
        elif basis_type == "gaussian":
            self.basis_fn = GaussianBasis(center, gamma, rc, n_basis)

        self.res_update = nn.Sequential(*[ResUpdate() for _ in range(depth)])
        gc_blocks = [PiNetGCBlock([], pi_nodes, ii_nodes, activation=act)]
        gc_blocks += [
            PiNetGCBlock(pp_nodes, pi_nodes, ii_nodes, activation=act)
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

        fc = self.cutoff(tensors["dist"])
        basis = self.basis_fn(tensors["dist"], fc=fc)
        output = 0.0
        for i in range(self.depth):
            p1 = self.gc_blocks[i](
                {
                    self.alias.ind_2: tensors["ind_2"],
                    self.alias.p1: tensors["p1"],
                    self.alias.diff: tensors["diff"],
                    self.alias.basis: basis,
                }
            )
            output = self.out_layers[i](p1, output)
            tensors["p1"] = self.res_update[i](tensors["p1"], p1)

        output = self.ann_output([tensors["ind_1"], output])
        return output
