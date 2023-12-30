# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-30
# version: 0.0.1

from collections import defaultdict
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import torch
import numpy as np

__all__ = ["Normalizer"]

class Statistic:

    def __init__(self):
        self._mean = 0
        self._stddev = 0
        self._count = 0
        self._M2 = 0

    def __call__(self, new):
        # assume new is a scalar or seq scalars
        if isinstance(new, torch.Tensor):
            new = new.numpy()
        else:
            new = np.atleast_1d(new)
        n = len(new)

        self._count += n
        delta = new - self._mean
        self._mean += np.sum(delta / n)
        delta2 = new - self._mean
        self._M2 += np.sum(delta * delta2)

    @property
    def mean(self):
        return self._mean
    
    @property
    def stddev(self):
        return (self._M2 / (self._count - 1)) ** 0.5
    
    @property
    def biased_stddev(self):
        return (self._M2 / self._count) ** 0.5

@functional_datapipe("normalize")
class Normalizer(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, keys: list):
        self.dp = source_dp
        self.keys = keys
        self._data = defaultdict(Statistic)

    def __iter__(self):
        for d in self.dp:
            for k in self.keys:
                self._data[k](d[k])
            d[k] = self._data[k].stddev * d[k] + self._data[k].mean
            
            yield d

    def __len__(self):
        return len(self.dp)
    