import pytest
import torch
from molpot import alias
from molpot.transforms import TorchNeighborList
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2 import DataLoader2

@pytest.fixture(params=[0])
def neighbor_list(request):
    neighbor_lists = [TorchNeighborList]
    return neighbor_lists[request.param]


class TestNeighborLists:
    """
    Test for different neighbor lists defined in neighbor_list using the Argon environment fixtures (periodic and
    non-periodic).

    """

    def test_neighbor_list(self, neighbor_list, environment):
        cutoff, input, neighbors_ref = environment
        neighbor_list = neighbor_list(cutoff)
        neighbors = neighbor_list(input)
        R = input[alias.R]
        neighbors[alias.Rij] = (
            R[neighbors[alias.idx_j]]
            - R[neighbors[alias.idx_i]]
            + input[alias.offsets]
        )

        neighbors = self._sort_neighbors(neighbors)
        neighbors_ref = self._sort_neighbors(neighbors_ref)

        for nbl, nbl_ref in zip(neighbors, neighbors_ref):
            torch.testing.assert_close(nbl, nbl_ref)

    def test_nblist_pipeline(self, neighbor_list, environment):

        cutoff, input, neighbors_ref = environment
        dp = IterableWrapper([input])
        dp = dp.calc_nblist(cutoff = cutoff)
        neighbors = next(iter(DataLoader2(dp)))
        R = input[alias.R]
        neighbors[alias.Rij] = (
            R[neighbors[alias.idx_j]]
            - R[neighbors[alias.idx_i]]
            + neighbors[alias.offsets]
        )

        neighbors = self._sort_neighbors(neighbors)
        neighbors_ref = self._sort_neighbors(neighbors_ref)

        for nbl, nbl_ref in zip(neighbors, neighbors_ref):
            torch.testing.assert_close(nbl, nbl_ref)

    def _sort_neighbors(self, neighbors):
        """
        Routine for sorting the index, shift and distance vectors to allow comparison between different
        neighbor list implementations.

        Args:
            neighbors: Input dictionary holding system neighbor information (idx_i, idx_j, cell_offset and Rij)

        Returns:
            torch.LongTensor: indices of central atoms in each pair
            torch.LongTensor: indices of each neighbor
            torch.LongTensor: cell offsets
            torch.Tensor: distance vectors associated with each pair
        """
        idx_i = neighbors[alias.idx_i]
        idx_j = neighbors[alias.idx_j]
        Rij = neighbors[alias.Rij]

        sort_idx = self._get_unique_idx(idx_i, idx_j, Rij)

        return idx_i[sort_idx], idx_j[sort_idx], Rij[sort_idx]

    @staticmethod
    def _get_unique_idx(
        idx_i: torch.Tensor, idx_j: torch.Tensor, offsets: torch.Tensor
    ):
        """
        Compute unique indices for every neighbor pair based on the central atom, the neighbor and the cell the
        neighbor belongs to. This is used for sorting the neighbor lists in order to compare between different
        implementations.

        Args:
            idx_i: indices of central atoms in each pair
            idx_j: indices of each neighbor
            offsets: cell offsets

        Returns:
            torch.LongTensor: indices used for sorting each tensor in a unique manner
        """
        n_max = torch.max(torch.abs(offsets))

        n_repeats = 2 * n_max + 1
        n_atoms = torch.max(idx_i) + 1

        unique_idx = (
            n_repeats**3 * (n_atoms * idx_i + idx_j)
            + (offsets[:, 0] + n_max)
            + n_repeats * (offsets[:, 1] + n_max)
            + n_repeats**2 * (offsets[:, 2] + n_max)
        )

        return torch.argsort(unique_idx)