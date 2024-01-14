# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-30
# version: 0.0.1

from collections import defaultdict, deque
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import torch
import numpy as np
from molpot.statistic.tracker import Tracker

__all__ = ["Normalizer"]


@functional_datapipe("normalize")
class Normalizer(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, keys: list):
        self.dp = source_dp
        self.keys = keys
        self._data = defaultdict(Tracker)

    def __iter__(self):
        for d in self.dp:
            for k in self.keys:
                self._data[k](d[k])
            d[k] = (d[k] - self._data[k].mean) / self._data[k].stddev
             
            yield d

    def __len__(self):
        return len(self.dp)
    
@functional_datapipe("atomic_dress")
class AtomicDressing(IterDataPipe):
    """
    http://mlwiki.org/index.php/Normal_Equation

    Args:
        IterDataPipe (_type_): _description_
    """
    def __init__(self, source_dp: IterDataPipe, types_list: list[int], key, prop, buffer:int|None = None):
        self.dp = source_dp
        self.types_list = torch.tensor(types_list)
        self.dress = defaultdict(Tracker)
        self.key = key
        self.prop = prop
        self.buffer = buffer

    def __iter__(self):

        x = deque(maxlen=self.buffer)
        y = deque(maxlen=self.buffer)
        for batch in self.dp:
            for sample in batch:
                atom_type = sample[self.key]
                target = sample[self.prop]
                count = torch.eq(
                    self.types_list.unsqueeze(1),
                    atom_type
                ).sum(dim=1)
                x.append(count)
                y.append(target)
            
            x_tensor = torch.stack(tuple(x))
            x_tensor = torch.cat((x_tensor, torch.zeros((x_tensor.shape[0], 1))), dim=1)
            y_tensor = torch.stack(tuple(y)).reshape(-1, 1)
            xTx = torch.matmul(x_tensor.T, x_tensor)
            xTx_inv = torch.linalg.pinv(xTx)
            xTx_invxT = torch.matmul(xTx_inv, x_tensor.T)
            w = torch.matmul(xTx_invxT, y_tensor)
            predict = x_tensor @ w
            # error = torch.sum((y_tensor - predict)**2)
            for i, sample in enumerate(batch):
                sample[self.prop] -= predict[i]

            yield batch
    
    def __len__(self):
        return len(self.dp)