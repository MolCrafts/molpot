 # author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-15
# version: 0.0.1

import functools

import numpy as np
import torch
from rich.console import Console
from rich.table import Column, Table
from ..pipeline.dataset import Dataset

class DataInspector:

    def __init__(self, dataset: Dataset):

        self.dataset = dataset

    def summary(self):

        console = Console()
        console.print(f"number of data: {self.dataset.total}")
        table = Table(title=f"dataset: {self.dataset.__class__.__name__}", )
        table.add_column("label", justify="center")
        table.add_column("type", justify="center")
        table.add_column("unit", justify="center")
        table.add_column("comment", justify="center")

        for label in self.dataset.labels.values():
            table.add_row(
                label.name,
                str(label.dtype),
                label.unit,
                label.comment
            )
        console.print(table)



    @functools.lru_cache
    def inspect(self, prop: str):

        data = []
        for sample in self.dataset:
            data.append(sample[prop])
        return torch.cat(data)
    
    def hist(self, prop: str):
        import matplotlib.pyplot as plt
        data = self.inspect(prop)
        d = data.detach().cpu().numpy()
        plt.hist(d, bins=100)
        plt.title(prop)
        plt.annotate(f"total: {len(d)}\nmean: {d.mean():.3f}\nstd: {d.std():.3f}", xy=(0.7, 0.7), xycoords="axes fraction")

        plt.show()
