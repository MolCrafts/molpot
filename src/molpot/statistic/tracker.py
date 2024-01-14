import numpy as np
import torch
from typing import TypeVar
number = TypeVar('number', int, float)

class Tracker:

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