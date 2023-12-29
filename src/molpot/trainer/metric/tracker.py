# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-03
# version: 0.1.0

from collections import defaultdict

__all__ = ["MetricTracker"]

class MetricTracker:
    def __init__(self, name:str, metrics:list):
        self.name = name
        self.metrics = []
        self.metrics.extend(metrics)
        self._total = defaultdict(float)
        self._counts = defaultdict(float)
        self._average = defaultdict(float)

    def __call__(self, step:int, output:dict, data:dict):
        for metric in self.metrics:
            value = metric(step, output, data)
            self._total[metric.name] += value * 1
            self._counts[metric.name] += 1
            self._average[metric.name] = self._total[metric.name] / self._counts[metric.name]

    @property
    def result(self):
        return dict(self._average)
    
    @property
    def metrics(self):
        return list(self._average.keys())