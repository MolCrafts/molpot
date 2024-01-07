# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-30
# version: 0.0.1

from collections import defaultdict
from typing import TypeVar
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import torch
import numpy as np

__all__ = ["Normalizer"]

number = TypeVar('number', int, float)

class Statistic:

    def __init__(self):
        self.reset()

    def __call__(self, new: number | np.ndarray | torch.Tensor):

        new = np.atleast_1d(new)
        n = len(new)

        self._count += n
        delta = new - self._mean
        self._mean += np.sum(delta / self._count)
        delta2 = new - self._mean
        self._M2 += np.sum(delta * delta2)

    def reset(self):
        self._mean = 0
        self._stddev = 0
        self._count = 0
        self._M2 = 0

    @property
    def mean(self):
        return self._mean
    
    @property
    def variance(self):
        return self._M2 / self._count
    
    @property
    def stddev(self):
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
            d[k] = (d[k] - self._data[k].mean) / self._data[k].stddev
            
            yield d

    def __len__(self):
        return len(self.dp)
    
@functional_datapipe("atomic_dress")
class AtomicDressing(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, atom_type: str, props: str):
        self.dp = source_dp
        self.atom_type = atom_type
        self.props = props
        self.dress = defaultdict(Statistic)

    def __iter__(self):
        for d in self.dp:

            atom_type = d[self.atom_type]
            y = d[self.props]
            unique_type, indices, x = atom_type.unique(return_counts=True, return_inverse=True)
            x_tensor = np.atleast_2d(np.array(x))
            y_tensor = np.atleast_2d(np.array(y))
            theta = np.dot(np.dot(np.linalg.pinv(np.dot(x_tensor.T, x_tensor)), x_tensor.T), y_tensor)
            dress = {e: float(theta[i]) for i, e in enumerate(unique_type)}
            # error = np.dot(x_tensor, theta) - y_tensor
            ave_dress = []
            for e, dre in dress.items():
                self.dress[str(e)](dre)
                ave_dress.append(self.dress[str(e)].mean)
            print(f"error: {torch.dot(x.double(), torch.tensor(ave_dress)) - y.double()}")
            d[self.props] -= torch.sum(torch.tensor(ave_dress)[indices])
            yield d
    
    def __len__(self):
        return len(self.dp)