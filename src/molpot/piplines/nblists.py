# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from torchdata.datapipes.iter import IterDataPipe
from molpot.transforms import TorchNeighborList
from torchdata.datapipes import functional_datapipe
import molpy as mp

__all__ = ["CalcNBList"]

@functional_datapipe("calc_nblist")
class CalcNBList(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, cutoff: float):
        self.dp = source_dp
        self.cutoff = cutoff

    def __iter__(self):
        nblist_fn = TorchNeighborList(cutoff=self.cutoff)
        for d in self.dp:
            assert isinstance(d, mp.Frame)
            nblist = nblist_fn(d)
            d
            yield d
