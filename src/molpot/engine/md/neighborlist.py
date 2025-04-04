import torch
from molpot import alias
from molpot_op import get_neighbor_pairs
from ignite.engine import Engine
from .event import MDEvents
from .handler import MDHandler


class NeighborList(MDHandler):
    """
    Wrapper for neighbor list transforms to make them suitable for molecular dynamics simulations. Introduces handling
    of multiple replicas and a cutoff shell (buffer region) to avoid recomputations of the neighbor list in every step.
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_skin: float,
        required_grad: bool = False,
        index_dtype: torch.dtype = torch.int32,
    ):
        """

        Args:
            cutoff (float): Cutoff radius.
            cutoff_skin (float): Buffer region. Atoms can move this much unitil neighbor list needs to be recomputed.
        """
        super().__init__({MDEvents.NEIGHBOR}, (0,))
        self.cutoff = cutoff
        self.cutoff_skin = cutoff_skin
        self.cutoff_full = cutoff + cutoff_skin
        self.required_grad = required_grad
        self.index_dtype = index_dtype

        self.previous_positions = None
        self.previous_cell = None

    def _update_required(self, positions: torch.Tensor, cell: torch.Tensor):
        """
        Use displacement and cell changes to determine, whether an update of the neighbor list is necessary.

        Args:
            positions (torch.Tensor): Atom positions.
            cell (torch.Tensor): Simulation cell.

        Returns:
            bool: Udate is required.
        """
        n_atoms = positions.shape[0]
        if self.previous_positions is None:
            # Everything needs to be updated
            update_required = torch.ones(n_atoms, device=positions.device).bool()
        else:
            # Check for changes is positions
            update_to_be = (
                torch.norm(self.previous_positions - positions, dim=1)
                > 0.5 * self.cutoff_skin
            ).bool()

            # Map to individual molecules
            update_required = torch.zeros(n_atoms, device=positions.device).float()
            update_required = update_required[update_to_be]

            # Check for cell changes (is no cell are required, this will always be zero)
            update_cell = torch.any(self.previous_cell != cell)
            update_required = torch.logical_or(update_required, update_cell)

        return update_required

    def on_neighbor(self, engine: Engine):
        """
        Compute neighbor indices from positions and simulations cell.

        Args:
            frame (dict(str, torch.Tensor)): input batch.

        Returns:
            torch.tensor: indices of neighbors.
        """
        frame = engine.state.frame
        positions = frame[alias.R]
        # molid = frame[alias.molid]
        if alias.cell not in frame:
            cell = None
        else:
            cell = frame[alias.cell]

        # Check which molecular environments need to be updated
        update_required = self._update_required(positions, cell)

        if torch.any(update_required):
            # if updated, store current positions and cell for future comparisons
            self.previous_positions = positions.clone()
            self.previous_cell = cell.clone()

            # calculate new neighbor list
            pairs, deltas, distances, n_pairs = get_neighbor_pairs(
                positions=positions, box_vectors=cell, cutoff=self.cutoff
            )
            pairs = pairs.to(dtype=self.index_dtype)

            mask = ~torch.isnan(distances)
            pairs = pairs[:, mask]
            deltas = deltas[mask]
            distances = distances[mask]
            n_pairs = mask.sum(dim=0)

            frame[alias.pair_i] = pairs[1]
            frame[alias.pair_j] = pairs[0]
            frame[alias.pair_diff] = deltas.requires_grad_(self.required_grad)
            frame[alias.pair_dist] = distances.requires_grad_(self.required_grad)

            frame[alias.n_pairs] = torch.tensor([n_pairs], dtype=torch.int32)
