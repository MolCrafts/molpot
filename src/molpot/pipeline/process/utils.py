from typing import TypeVar

import torch

number = TypeVar('number', int, float)

class Tracker:
    """
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._mean = 0
        self._stddev = 0
        self._count = 0
        self._M2 = 0

    def __call__(self, new: number | torch.Tensor):

        self.update(new)

    def update(self, new: number | torch.Tensor):
        """
        update tracker with a new scalar or a tensor of scalars

        Args:
            new (number | torch.Tensor): new scalar or tensor of scalars
        """
        if not torch.is_tensor(new):
            new = torch.atleast_1d(torch.tensor(new))
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
    def sample_variance(self):
        return self._M2 / (self._count - 1) if self._count > 1 else 0