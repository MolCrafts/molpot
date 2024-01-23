# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from torchdata.datapipes.iter import IterDataPipe
from molpot.transforms import TorchNeighborList
from torchdata.datapipes import functional_datapipe
from molpot import alias, Config

__all__ = ["CalcNBList"]

@functional_datapipe("calc_nblist")
class CalcNBList(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, cutoff: float):
        self.dp = source_dp
        self.cutoff = cutoff
        self.kernel = TorchNeighborList(cutoff).to(Config.device)

    def __iter__(self):
        for d in self.dp:
            yield [self.kernel(dd) for dd in d]

    def __len__(self):
        return len(self.dp)
