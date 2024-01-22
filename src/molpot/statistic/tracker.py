import torch
from typing import TypeVar
number = TypeVar('number', int, float)

class Tracker:

    def __init__(self):
        self._mean = 0
        self._stddev = 0
        self._count = 0
        self._M2 = 0

    def __call__(self, new: number | torch.Tensor):

        new = torch.atleast_1d(new)
        n = len(new)

        self._count += n
        delta = new - self._mean
        self._mean += torch.sum(delta / self._count)
        delta2 = new - self._mean
        self._M2 += torch.sum(delta * delta2)

    @property
    def mean(self):
        return self._mean
    
    @property
    def variance(self):
        return self._M2 / self._count
    
    @property
    def stddev(self):
        return self.variance ** 0.5