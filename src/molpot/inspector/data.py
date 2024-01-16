# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-15
# version: 0.0.1

from collections import defaultdict
import torch
import numpy as np

__all__ = ["DataInspector"]


class DataInspector:

    def __init__(self, dataloader):

        self.dataloader = dataloader

    def inspect(self, prop: str, nbatch:int = 0):

        data = []

        for i, sample in enumerate(self.dataloader, 1):
            data.append(sample[prop])
            if i == nbatch:
                break
        return torch.stack(data).flatten()
    
    def distribute(self, prop: str, nbatch:int = 0):
        import matplotlib.pyplot as plt
        data = self.inspect(prop, nbatch)
        d = data.detach().cpu().numpy()
        plt.hist(d, bins=100)
        plt.title(prop)
        plt.annotate(f"total: {len(d)}\nmean: {np.mean(d):.3f}\nstd: {np.std(d):.3f}", xy=(0.7, 0.7), xycoords="axes fraction")

        plt.show()
