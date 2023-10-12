# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-10-12
# version: 0.0.1

import torch
from torch import nn

class FFLayer(nn.modules):

    def __init__(self, n_nodes=[64, 64], **kwargs):

        super().__init__()
        self.dense_layers = nn.Sequential(
            *[nn.Linear(n_node) for n_node in n_nodes]
        )

    def forward(self, x):
        return self.dense_layers(x)
    
class PILayer(nn.Module):
    def __init__(self, n_nodes=[64], n_basis=4, **kwargs):
        super(PILayer, self).__init__()
        self.n_nodes = n_nodes
        self.kwargs = kwargs
        self.n_basis = None
        self.ff_layer = None
        self.n_basis = n_basis

        n_nodes_iter = self.n_nodes.copy()
        n_nodes_iter[-1] *= self.n_basis
        self.ff_layer = nn.Sequential(
            *[nn.Linear(in_features, out_features, **self.kwargs) for in_features, out_features in zip(n_nodes_iter, n_nodes_iter[1:])]
        )

    def forward(self, tensors):
        ind_2, prop, basis = tensors
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        prop_i = prop[ind_i]
        prop_j = prop[ind_j]

        inter = torch.cat([prop_i, prop_j], dim=-1)
        inter = self.ff_layer(inter)
        inter = inter.view(-1, self.n_nodes[-1], self.n_basis)
        inter = torch.einsum("pcb,pb->pc", [inter, basis])
        return inter
    
class PIXLayer(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super(PIXLayer, self).__init__()

    def forword(self, tensors):
        """
        PILayer take a list of three tensors as input:

        - ind_2: [sparse indices](layers.md#sparse-indices) of pairs with shape `(n_pairs, 2)`
        - prop: property tensor with shape `(n_atoms, n_prop)`

        Args:
            tensors (list of tensors): list of `[ind_2, prop, basis]` tensors

        Returns:
            inter (tensor): interaction tensor with shape `(n_pairs, n_nodes[-1])`
        """
        ind_2, px = tensors
        ind_i = ind_2[:, 0]
        ind_j = ind_2[:, 1]
        # px_i = px[ind_i]
        px_j = px[ind_j]
       
        return px_j
    
class DotLayer(nn.modules):

    def __init__(self, **kwargs):
        """
        Args:
            style (str): style of the layer, should be one of 'painn', 'newton', 'general'
        """
        super(DotLayer, self).__init__()

    def forword(self, tensor):

        return torch.einsum("ixr,ixr->ir", tensor, tensor)
    
class ScaleLayer(nn.modules):

    def __init__(self, **kwargs):

        super(ScaleLayer, self).__init__()

    def __call__(self, tensor):

        px, p1 = tensor
        return px * p1[:, None, :]
    
class IPLayer(nn.modules):

    def __init__(self):

        super(IPLayer, self).__init__()

    def call(self, tensors):

        ind_2, prop, inter = tensors
        return torch.scatter_add(torch.zeros_like(prop.shape[0]), 0, ind_2[:, 0], inter)
    
class OutLayer(nn.modules):

    def __init__(self, n_nodes, out_nodes, **kwargs):

        super(OutLayer, self).__init__()
        self.n_nodes = n_nodes
        self.out_nodes = out_nodes
        self.ff_layer = FFLayer(n_nodes, **kwargs)
        self.out_layer = nn.Linear(n_nodes, bias=False)

    def forward(self, tensors):

        ind_1, p1, p3, prev_output = tensors
        p1 = self.ff_layer(p1)
        output = self.out_layer(p1) + prev_output
        return output
    
class GCBlock(nn.modules):

    def __init__(self, pp_nodes, pi_nodes, ii_nodes, **kwargs):
        super().__init__()
        iiargs = kwargs.copy()
        iiargs.update({"bias": False})
        ii_nodes = ii_nodes.copy()
        ii_nodes[-1] *= 3
        self.pp1_layer = FFLayer(pp_nodes, **kwargs)
        self.pi1_layer = PILayer(pi_nodes, **kwargs)
        self.ii1_layer = FFLayer(ii_nodes, **iiargs)
        self.ip1_layer = IPLayer()

        self.pix_layer = PIXLayer()
        self.ii3_layer = FFLayer(ii_nodes, **iiargs)
        self.ip3_layer = IPLayer()

        self.dot_layer = DotLayer()

        self.scale1_layer = ScaleLayer()
        self.scale2_layer = ScaleLayer()
        self.scale3_layer = ScaleLayer()

    def forword(self, tensors):

        ind_2, p1, p3, diff, basis = tensors

        p1 = self.pp1_layer(p1)
        i1 = self.pi1_layer([ind_2, p1, basis])
        i1 = self.ii1_layer(i1)
        i1_1, i1_2, i1_3 = torch.split(i1, 3, axis=-1)
        p1 = self.ip1_layer([ind_2, p1, i1_2])


        i3 = self.pix_layer([ind_2, p3])
        i3 = self.scale1_layer([i3, i1_3])
        scaled_diff = self.scale2_layer([diff[:,:,None], i1_3])
        i3 = i3 + scaled_diff
        p3 = self.ip3_layer([ind_2, p3, i3])

        p1t1 = self.dot_layer(p3) + p1
        p3t1 = self.scale3_layer([p3, p1t1])

        return p1t1, p3t1