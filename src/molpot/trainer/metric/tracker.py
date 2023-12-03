# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-03
# version: 0.1.0

from collections import defaultdict

__all__ = ["MetricTracker"]

class MetricTracker:
    def __init__(self, name:str):
        self.name = name
        self.reset()

    def reset(self):
        self._total = defaultdict(float)
        self._counts = defaultdict(float)
        self._average = defaultdict(float)

    def update(self, key, value, n=1):

        self._total[key] += value * n
        self._counts[key] += n
        self._average[key] = self._total[key] / self._counts[key]

    @property
    def result(self):
        return dict(self._average)
    
    @property
    def metrics(self):
        return list(self._average.keys())