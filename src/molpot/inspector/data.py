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

    def inspect(self, props: list, nbatch:int = 0):

        data = defaultdict(list)

        for i, batch in enumerate(self.dataloader, 1):
            for prop in props:
                data[prop].extend(batch[prop])
            if i == nbatch:
                break
        return data
    
    def distribute(self, props: list, nbatch:int = 0):
        import matplotlib.pyplot as plt
        data = self.inspect(props, nbatch)
        nprops = len(props)

        fig, axs = plt.subplots(nprops, 1, figsize=(6*nprops, 6))
        if isinstance(axs, plt.Axes):
            axs = [axs]
        for i, prop in enumerate(props):
            d = np.array(data[prop])
            axs[i].hist(d, bins=100)
            axs[i].set_title(prop)
            axs[i].annotate(f"total: {len(d)}\nmean: {np.mean(d):.3f}\nstd: {np.std(d):.3f}", xy=(0.7, 0.7), xycoords="axes fraction")

        plt.show()
