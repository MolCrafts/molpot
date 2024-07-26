# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from molpot import Config

import numpy as np
import torch
from torch import nn

from molpot import alias
class TorchNeighborList(nn.Module):
    """
    Environment provider making use of neighbor lists as implemented in TorchAni

    Supports cutoffs and PBCs and can be performed on either CPU or GPU.

    References:
        https://github.com/aiqm/torchani/blob/master/torchani/aev.py
    """

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
        inputs: dict,
    ) -> dict:
        xyz = inputs[alias.xyz]
        cell = inputs[alias.cell]
        pbc = inputs[alias.pbc]

        pair_i, pair_j, offset = self._build_neighbor_list(
            xyz, cell, pbc, self._cutoff
        )
        inputs[alias.pair_i] = pair_i.detach()
        inputs[alias.pair_j] = pair_j.detach()
        inputs[alias.offsets] = offset

        return inputs

    def _build_neighbor_list(self, positions, cell, pbc, cutoff):
        # Check if shifts are needed for periodic boundary conditions
        if all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device, dtype=torch.long)
        else:
            shifts = self._get_shifts(cell, pbc, cutoff)
        pair_i, pair_j, offset = self._get_neighbor_pairs(
            positions, cell, shifts, cutoff
        )

        # Create bidirectional id arrays, similar to what the ASE neighbor_list returns
        bi_idx_i = torch.cat((pair_i, pair_j), dim=0)
        bi_idx_j = torch.cat((pair_j, pair_i), dim=0)

        # Sort along first dimension (necessary for atom-wise pooling)
        sorted_idx = torch.argsort(bi_idx_i)
        pair_i = bi_idx_i[sorted_idx]
        pair_j = bi_idx_j[sorted_idx]

        bi_offset = torch.cat((-offset, offset), dim=0)
        offset = bi_offset[sorted_idx]
        offset = torch.mm(offset.to(cell.dtype), cell)

        return pair_i, pair_j, offset

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
        cutoff_mask = distances < cutoff
        in_cutoff = torch.nonzero(cutoff_mask, as_tuple=False)

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

class PairwiseDistances(nn.Module):
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xyz = inputs[alias.xyz]
        offsets = inputs[alias.offsets]
        pair_i = inputs[alias.pair_i]
        pair_j = inputs[alias.pair_j]

        # To avoid error in Windows OS
        pair_i = pair_i.long()
        pair_j = pair_j.long()

        inputs[alias.d_ij] = xyz[pair_j] - xyz[pair_i] + offsets
        return inputs


class FilterShortRange(nn.Module):
    """
    Separate short-range from all supplied distances.

    The short-range distances will be stored under the original keys (dl_ij,
    pair_i, pair_j), while the original distances can be accessed for long-range terms via
    (dl_ij, pair_i, idx_j_lr).
    """

    def __init__(self, short_range_cutoff: float):
        super().__init__()
        self.short_range_cutoff = short_range_cutoff

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pair_i = inputs[pair_i]
        pair_j = inputs[pair_j]
        Rij = inputs[Rij]

        rij = torch.norm(Rij, dim=-1)
        cidx = torch.nonzero(rij <= self.short_range_cutoff).squeeze(-1)

        inputs[alias.dl_ij] = Rij
        inputs[alias.pair_i_lr] = pair_i
        inputs[alias.pair_j_lr] = pair_j

        inputs[alias.Rij] = Rij[alias.cidx]
        inputs[alias.pair_i] = pair_i[alias.cidx]
        inputs[alias.pair_j] = pair_j[alias.cidx]
        return inputs

class NeighborsFilter(nn.Module):
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
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        n_neighbors = inputs[alias.pair_i].shape[0]
        slab_indices = inputs[self.selection_name].tolist()
        kept_nbh_indices = []
        for nbh_idx in range(n_neighbors):
            i = inputs[alias.pair_i][nbh_idx].item()
            j = inputs[alias.pair_j][nbh_idx].item()
            if i not in slab_indices or j not in slab_indices:
                kept_nbh_indices.append(nbh_idx)

        inputs[alias.pair_i] = inputs[alias.pair_i][kept_nbh_indices]
        inputs[alias.pair_j] = inputs[alias.pair_j][kept_nbh_indices]
        inputs[alias.offsets] = inputs[alias.offsets][kept_nbh_indices]

        return inputs


class TripleGenerator(nn.Module):
    """
    Generate the index tensors for all triples between atoms within the cutoff shell.
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Using the neighbors contained within the cutoff shell, generate all unique pairs
        of neighbors and convert them to index arrays. Applied to the neighbor arrays,
        these arrays generate the indices involved in the atom triples.

        Example:
            pair_j[idx_j_triples] -> j atom in triple
            pair_j[idx_k_triples] -> k atom in triple
            Rij[idx_j_triples] -> Rij vector in triple
            Rij[idx_k_triples] -> Rik vector in triple
        """
        pair_i = inputs[pair_i]

        _, n_neighbors = torch.unique_consecutive(pair_i, return_counts=True)

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

        inputs[idx_i_triples] = idx_i_triples
        inputs[idx_j_triples] = idx_j_triples.squeeze(-1)
        inputs[idx_k_triples] = idx_k_triples.squeeze(-1)
        return inputs


class NeighborsCounter(nn.Module):
    """
    Store the number of neighbors for each atom
    """

    is_preprocessor: bool = True
    is_postprocessor: bool = False

    def __init__(self, sorted: bool = True):
        """
        Args:
            sorted: Set to false if chosen neighbor list yields unsorted center indices
                (pair_i).
        """
        super(NeighborsCounter, self).__init__()
        self.sorted = sorted

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        pair_i = inputs[pair_i]

        if self.sorted:
            _, n_nbh = torch.unique_consecutive(pair_i, return_counts=True)
        else:
            _, n_nbh = torch.unique(pair_i, return_counts=True)

        inputs[n_nbh] = n_nbh
        return inputs


class PositionWrapper(nn.Module):
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
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        xyz = inputs[xyz]
        cell = inputs[cell].view(3, 3)
        pbc = inputs[pbc]

        inverse_cell = torch.inverse(cell)
        inv_positions = torch.sum(xyz[..., None] * inverse_cell[None, ...], dim=1)

        periodic = torch.masked_select(inv_positions, pbc[None, ...])

        # Apply periodic boundary conditions (with small buffer)
        periodic = periodic + self.eps
        periodic = periodic % 1.0
        periodic = periodic - self.eps

        # Update fractional coordinates
        inv_positions.masked_scatter_(pbc[None, ...], periodic)

        # Convert to positions
        R_wrapped = torch.sum(inv_positions[..., None] * cell[None, ...], dim=1)

        inputs[xyz] = R_wrapped

        return inputs



@functional_datapipe("calc_nblist")
class CalcNBList(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, cutoff: float):
        self.dp = source_dp
        self.cutoff = cutoff
        self.kernel = TorchNeighborList(cutoff).to(Config.device)

    def __iter__(self):
        for d in self.dp:
            yield self.kernel(d)

    def __len__(self):
        return len(self.dp)

@functional_datapipe("calc_dist")
class CalcDist(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe):
        self.dp = source_dp
        self.kernel = PairwiseDistances().to(Config.device)

    def __iter__(self):
        for d in self.dp:
            yield self.kernel(d)

    def __len__(self):
        return len(self.dp)

@functional_datapipe("filter_dist")
class FilterDist(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, short_cutoff: float):
        self.dp = source_dp
        self.short_cutoff = short_cutoff
        self.kernel = FilterShortRange(short_cutoff).to(Config.device)

    def __iter__(self):
        for d in self.dp:
            yield [self.kernel(dd) for dd in d]

    def __len__(self):
        return len(self.dp)