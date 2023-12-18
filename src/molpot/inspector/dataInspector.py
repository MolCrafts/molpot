# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-15
# version: 0.0.1

from collections import defaultdict
import torch

__all__ = ["DataInspector"]


class DataInspector:

    def __init__(self, dataloader):

        self.dataloader = dataloader

    def inspect(self, nbatch: int, props: list):

        data = defaultdict(list)

        for i, batch in enumerate(self.dataloader):
            for prop in props:
                data[prop].append(batch[prop])
            if i == nbatch:
                break
        return data
