import os
import torch
import shutil
from .base import Transform
# from dirsync import sync
import numpy as np
from typing import Optional, Dict, List
import molpy as mp

__all__ = [
    "TorchNeighborList",
    "NeighborsCounter",
    "TripleGenerator",
    "CachedNeighborList",
    "NeighborListTransform",
    "PositionWrapper",
    "BufferNeighborList",
    "NeighborsFilter",
]

import molpot as mpot


class NeighborListTransform(Transform):
    """
    Base class for neighbor lists.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(
        self,
        cutoff: float,
    ):
        """
        Args:
            cutoff: Cutoff radius for neighbor search.
        """
        super().__init__()
        self._cutoff = cutoff

    def forward(
        self,
        inputs: mp.Frame,
    ) -> mp.Frame:
        R = inputs.atoms[mpot.alias.R]
        cell = inputs.box.matrix
        pbc = inputs.box.pbc

        idx_i, idx_j, offset = self._build_neighbor_list(R, cell, pbc, self._cutoff)
        inputs.atoms[mpot.alias.idx_i] = idx_i.detach()
        inputs.atoms[mpot.alias.idx_j] = idx_j.detach()
        inputs.atoms[mpot.alias.offsets] = offset
        return inputs

    def _build_neighbor_list(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
    ):
        """Override with specific neighbor list implementation"""
        raise NotImplementedError


class TorchNeighborList(NeighborListTransform):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni

    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    References:
        https://github.com/aiqm/torchani/blob/master/torchani/aev.py
    """

    def _build_neighbor_list(self, positions, cell, pbc, cutoff):

        cell = torch.tensor(cell, dtype=torch.float32)
        pbc = torch.tensor(pbc, dtype=torch.bool)

        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device, dtype=torch.long)
        else:
            shifts = self._get_shifts(cell, pbc, cutoff)
        idx_i, idx_j, offset = self._get_neighbor_pairs(positions, cell, shifts, cutoff)

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((idx_i, idx_j), dim=0)
        bi_idx_j = torch.cat((idx_j, idx_i), dim=0)

        # Sort along first dimension (necessary for atom-wise pooling)
        sorted_idx = torch.argsort(bi_idx_i)
        idx_i = bi_idx_i[sorted_idx]
        idx_j = bi_idx_j[sorted_idx]

        bi_offset = torch.cat((-offset, offset), dim=0)
        offset = bi_offset[sorted_idx]
        offset = torch.mm(offset.to(cell.dtype), cell)

        return idx_i, idx_j, offset

    def _get_neighbor_pairs(self, positions, cell, shifts, cutoff):
        """Compute pairs of atoms that are neighbors
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        # 1) Central cell
        pi_center, pj_center = torch.combinations(all_atoms).unbind(-1)
        shifts_center = shifts.new_zeros(pi_center.shape[0], 3)

        # 2) cells with shifts
        # shape convention (shift index, molecule index, atom index, 3)
        num_shifts = shifts.shape[0]
        all_shifts = torch.arange(num_shifts, device=cell.device)
        shift_index, pi, pj = torch.cartesian_prod(
            all_shifts, all_atoms, all_atoms
        ).unbind(-1)
        shifts_outside = shifts.index_select(0, shift_index)

        # 3) combine results for all cells
        shifts_all = torch.cat([shifts_center, shifts_outside])
        pi_all = torch.cat([pi_center, pi])
        pj_all = torch.cat([pj_center, pj])

        # 4) Compute shifts and distance vectors
        shift_values = torch.mm(shifts_all.to(cell.dtype), cell)
        Rij_all = positions[pi_all] - positions[pj_all] + shift_values

        # 5) Compute distances, and find all pairs within cutoff
        distances = torch.norm(Rij_all, dim=1)
        in_cutoff = torch.nonzero(distances < cutoff, as_tuple=False)

        # 6) Reduce tensors to relevant components
        pair_index = in_cutoff.squeeze()  # potential bug if in_cutoff.shape == (1, 1)
        atom_index_i = pi_all[pair_index]
        atom_index_j = pj_all[pair_index]
        offsets = shifts_all[pair_index]

        return atom_index_i, atom_index_j, offsets

    def _get_shifts(self, cell, pbc, cutoff):
        """Compute the shifts of unit cell along the given cell vectors to make it
        large enough to contain all pairs of neighbor atoms with PBC under
        consideration.
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
                if pbc is enabled for that direction.
        Returns:
            :class:`torch.Tensor`: long tensor of shifts. the center cell and
                symmetric cells are not included.
        """
        reciprocal_cell = cell.inverse().t()
        inverse_lengths = torch.norm(reciprocal_cell, dim=1)

        num_repeats = torch.ceil(cutoff * inverse_lengths).long()
        num_repeats = torch.where(
            pbc, num_repeats, torch.Tensor([0], device=cell.device).long()
        )

        r1 = torch.arange(1, num_repeats[0] + 1, device=cell.device)
        r2 = torch.arange(1, num_repeats[1] + 1, device=cell.device)
        r3 = torch.arange(1, num_repeats[2] + 1, device=cell.device)
        o = torch.zeros(1, dtype=torch.long, device=cell.device)

        return torch.cat(
            [
                torch.cartesian_prod(r1, r2, r3),
                torch.cartesian_prod(r1, r2, o),
                torch.cartesian_prod(r1, r2, -r3),
                torch.cartesian_prod(r1, o, r3),
                torch.cartesian_prod(r1, o, o),
                torch.cartesian_prod(r1, o, -r3),
                torch.cartesian_prod(r1, -r2, r3),
                torch.cartesian_prod(r1, -r2, o),
                torch.cartesian_prod(r1, -r2, -r3),
                torch.cartesian_prod(o, r2, r3),
                torch.cartesian_prod(o, r2, o),
                torch.cartesian_prod(o, r2, -r3),
                torch.cartesian_prod(o, o, r3),
            ]
        )

class NeighborsFilter(Transform):
    """
    Filter out all neighbor list indices corresponding to interactions between a set of
    atoms. This set of atoms must be specified in the input data.
    """

    def __init__(self, selection_name: str):
        """
        Args:
            selection_name (str): key in the input data corresponding to the set of
                atoms between which no interactions should be considered.
        """
        self.selection_name = selection_name
        super().__init__()

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        n_neighbors = inputs[mpot.alias.idx_i].shape[0]
        slab_indices = inputs[self.selection_name].tolist()
        kept_nbh_indices = []
        for nbh_idx in range(n_neighbors):
            i = inputs[mpot.alias.idx_i][nbh_idx].item()
            j = inputs[mpot.alias.idx_j][nbh_idx].item()
            if i not in slab_indices or j not in slab_indices:
                kept_nbh_indices.append(nbh_idx)

        inputs[mpot.alias.idx_i] = inputs[mpot.alias.idx_i][kept_nbh_indices]
        inputs[mpot.alias.idx_j] = inputs[mpot.alias.idx_j][kept_nbh_indices]
        inputs[mpot.alias.offsets] = inputs[mpot.alias.offsets][kept_nbh_indices]

        return inputs

class TripleGenerator(Transform):
    """
    Generate the index tensors for all triples between atoms within the cutoff shell.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Using the neighbors contained within the cutoff shell, generate all unique pairs
        of neighbors and convert them to index arrays. Applied to the neighbor arrays,
        these arrays generate the indices involved in the atom triples.

        Example:
            idx_j[idx_j_triples] -> j atom in triple
            idx_j[idx_k_triples] -> k atom in triple
            Rij[idx_j_triples] -> Rij vector in triple
            Rij[idx_k_triples] -> Rik vector in triple
        """
        idx_i = inputs[mpot.alias.idx_i]

        _, n_neighbors = torch.unique_consecutive(idx_i, return_counts=True)

        offset = 0
        idx_i_triples = ()
        idx_jk_triples = ()
        for idx in range(n_neighbors.shape[0]):
            triples = torch.combinations(
                torch.arange(offset, offset + n_neighbors[idx]), r=2
            )
            idx_i_triples += (torch.ones(triples.shape[0], dtype=torch.long) * idx,)
            idx_jk_triples += (triples,)
            offset += n_neighbors[idx]

        idx_i_triples = torch.cat(idx_i_triples)

        idx_jk_triples = torch.cat(idx_jk_triples)
        idx_j_triples, idx_k_triples = idx_jk_triples.split(1, dim=-1)

        inputs[mpot.alias.idx_i_triples] = idx_i_triples
        inputs[mpot.alias.idx_j_triples] = idx_j_triples.squeeze(-1)
        inputs[mpot.alias.idx_k_triples] = idx_k_triples.squeeze(-1)
        return inputs


class NeighborsCounter(Transform):
    """
    Store the number of neighbors for each atom
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, sorted: bool = True):
        """
        Args:
            sorted: Set to false if chosen neighbor list yields unsorted center indices
                (idx_i).
        """
        super(NeighborsCounter, self).__init__()
        self.sorted = sorted

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        idx_i = inputs[mpot.alias.idx_i]

        if self.sorted:
            _, n_nbh = torch.unique_consecutive(idx_i, return_counts=True)
        else:
            _, n_nbh = torch.unique(idx_i, return_counts=True)

        inputs[mpot.alias.n_nbh] = n_nbh
        return inputs


class PositionWrapper(Transform):
    """
    Wrap atom positions into periodic cell. This routine requires a non-zero cell.
    The cell center of the inverse cell is set to (0.5, 0.5, 0.5).
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps (float): small offset for numerical stability.
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        R = inputs[mpot.alias.R]
        cell = inputs[mpot.alias.cell].view(3, 3)
        pbc = inputs[mpot.alias.pbc]

        inverse_cell = torch.inverse(cell)
        inv_positions = torch.sum(R[..., None] * inverse_cell[None, ...], dim=1)

        periodic = torch.masked_select(inv_positions, pbc[None, ...])

        # Apply periodic boundary conditions (with small buffer)
        periodic = periodic + self.eps
        periodic = periodic % 1.0
        periodic = periodic - self.eps

        # Update fractional coordinates
        inv_positions.masked_scatter_(pbc[None, ...], periodic)

        # Convert to positions
        R_wrapped = torch.sum(inv_positions[..., None] * cell[None, ...], dim=1)

        inputs[mpot.alias.R] = R_wrapped

        return inputs