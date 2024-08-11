# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from molpot import Config

from molpot.utils import TorchNeighborList, PairwiseDistances

@functional_datapipe("calc_nblist")
class CalcNBList(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, cutoff: float):
        self.dp = source_dp
        self.cutoff = cutoff
        self.kernel = TorchNeighborList(cutoff)

    def __iter__(self):
        for d in self.dp:
            yield self.kernel(d)

    def __len__(self):
        return len(self.dp)


@functional_datapipe("calc_dist")
class CalcDist(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe):
        self.dp = source_dp
        self.kernel = PairwiseDistances()

    def __iter__(self):
        for d in self.dp:
            yield self.kernel(d)

    def __len__(self):
        return len(self.dp)
