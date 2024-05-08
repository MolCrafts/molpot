import torch
import torch.nn as nn
import torch.nn.functional as F
from molpy import Alias
from .block import build_mlp
from typing import Callable
from torch_scatter import scatter_add

class PILayer(nn.Module):

    def __init__(self, n_nodes: list[int], activation: Callable | None = F.tanh):
        super().__init__()
        self.n_nodes = n_nodes
        self.mlp = build_mlp(n_nodes[0], n_nodes[-1], n_nodes[1:-1], None, activation)

    def forward(self, idx_i, idx_j, prop, basis):

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

class GCBlock(nn.Module):
    
    def __init__(self, pp_nodes: list[int], pi_nodes: list[int], ii_nodes: list[int], activation: Callable | None = F.tanh):
        super().__init__()

        # Scalar property
        self.pp1_layer = build_mlp(pp_nodes[0], pp_nodes[-1], pp_nodes[1:-1], None, activation)
        self.pi1_layer = PILayer(pi_nodes, activation)
        self.ii1_layer = build_mlp(ii_nodes[0], ii_nodes[-1], ii_nodes[1:-1], None, activation)
        self.ip1_layer = IPLayer()

    def forward(self, idx_i, idx_j, p1, basis) -> dict[str, torch.Tensor]:

        pp1 = self.pp1_layer(p1)
        pi1 = self.pi1_layer(idx_i, idx_j, pp1, basis)
        ii1 = self.ii1_layer(pi1)
        ip1 = self.ip1_layer(idx_i, ii1)

        return ip1
    
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
        rank: int = 1
    ):
        super().__init__()

        Alias('pinet')
        Alias.pinet.set('p1', 'pinet_p1', torch.Tensor, None, "Scalar property")
        Alias.pinet.set('p3', 'pinet_p3', torch.Tensor, None, "Vectorial property")

        self.depth = depth
        self.basis_fn = basis_fn
        self.cutoff_fn = cutoff_fn

        pp_nodes = [n_atom_basis] + pp_nodes

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.basis_transformation = nn.Linear(n_atom_basis, pp_nodes[-1])

        self.res_update = nn.ModuleList(
            [ResUpdate() for _ in range(depth - 1)]
        )
        self.gc1_blocks = nn.ModuleList(
            [GCBlock(pp_nodes, pi_nodes, ii_nodes, activation=activation) for _ in range(depth)]
        )
        self.out_nodes = pp_nodes[-1]

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        # get tensors from input dictionary
        atomic_numbers = inputs[Alias.Z]
        n_atoms = atomic_numbers.shape[0]
        R = inputs[Alias.R]
        idx_i = inputs[Alias.idx_i]
        idx_j = inputs[Alias.idx_j]
        offsets = inputs[Alias.offsets]
        r_ij = R[idx_j] - R[idx_i] + offsets
        d_ij = torch.norm(r_ij, dim=-1, keepdim=True)
        dir_ij = r_ij / d_ij

        basis = self.basis_fn(d_ij)
        fc = self.cutoff_fn(d_ij)
        phi = self.basis_transformation(basis * fc[..., None])
        # (n_atoms, n_dims, n_atom_basis)
        inputs['p1'] = self.embedding(atomic_numbers)[:, None, :]  
        
        inputs["p1"] = self.gc1_blocks[0](idx_i, idx_j, inputs['p1'], phi)
        for i in range(self.depth, 1):
            p1 = self.gc1_blocks[i](idx_i, idx_j, inputs['p1'], phi)
            inputs['p1'] = self.res_update[i-1](inputs['p1'], p1)

        return inputs