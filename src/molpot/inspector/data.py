 # author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-15
# version: 0.0.1

from collections import defaultdict
import functools

import numpy as np
import torch

__all__ = ["DataInspector"]


class DataInspector:

    def __init__(self, dataloader):

        self.dataloader = dataloader

    @functools.lru_cache
    def inspect(self, prop: str):

        data = []
        for sample in self.dataloader:
            data.append(sample[prop])
        return torch.cat(data).flatten()
    
    def plot_dist(self, prop: str):
        import matplotlib.pyplot as plt
        data = self.inspect(prop)
        d = data.detach().cpu().numpy()
        plt.hist(d, bins=100)
        plt.title(prop)
        plt.annotate(f"total: {len(d)}\nmean: {d.mean():.3f}\nstd: {d.std():.3f}", xy=(0.7, 0.7), xycoords="axes fraction")

        plt.show()
