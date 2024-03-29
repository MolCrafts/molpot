import torch
from torch import nn

class FilterShortRange(nn.Module):
    """
    Separate short-range from all supplied distances.

    The short-range distances will be stored under the original keys (properties.Rij,
    properties.idx_i, properties.idx_j), while the original distances can be accessed for long-range terms via
    (properties.Rij_lr, properties.idx_i_lr, properties.idx_j_lr).
    """

    def __init__(self, short_range_cutoff: float):
        super().__init__()
        self.short_range_cutoff = short_range_cutoff

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        idx_i = inputs[kw.idx_i]
        idx_j = inputs[kw.idx_j]
        Rij = inputs[kw.Rij]

        rij = torch.norm(Rij, dim=-1)
        cidx = torch.nonzero(rij <= self.short_range_cutoff).squeeze(-1)

        inputs[kw.Rij_lr] = Rij
        inputs[kw.idx_i_lr] = idx_i
        inputs[kw.idx_j_lr] = idx_j

        inputs[kw.Rij] = Rij[cidx]
        inputs[kw.idx_i] = idx_i[cidx]
        inputs[kw.idx_j] = idx_j[cidx]
        return inputs