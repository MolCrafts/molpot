from ..base import Fix
from molpot.utils import TorchNeighborList, PairwiseDistances

class NaiveNeighborList(Fix):

    def __init__(self, cutoff:float, every_n_steps: int=1):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.nblist_kernel = TorchNeighborList(cutoff)
        self.dist_kernel = PairwiseDistances()

    def forward(self, engine, status, inputs, outputs):
        if status['current_step'] % self.every_n_steps == 0:
            self.nblist_kernel(inputs, outputs)
            self.dist_kernel(inputs, outputs)
        return inputs, outputs