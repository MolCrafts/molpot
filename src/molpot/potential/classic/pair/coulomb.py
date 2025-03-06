from molpot_op import PMEkernel
import torch

class PME(torch.nn.Module):
    def __init__(
        self,
        cutoff: float,
        gridx: int,
        gridy: int,
        gridz: int,
        order: int,
        alpha: float,
        coulomb: float,
        exclusions: torch.Tensor,
    ):
        super().__init__()
        self.kernel = PMEkernel(gridx, gridy, gridz, order, alpha, coulomb, exclusions)
        self.cutoff = cutoff

    def forward(self, positions: torch.Tensor, charges: torch.Tensor, box_vectors: torch.Tensor, pairs: torch.Tensor|None=None):
        edir = self.kernel.compute_direct(positions, charges, self.cutoff, box_vectors, pairs, max_num_pairs=-1)
        erecip = self.kernel.compute_reciprocal(positions, charges, box_vectors)
        return edir + erecip
