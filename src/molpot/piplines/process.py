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
                unique_type, indices, count = atom_type.unique(return_counts=True, return_inverse=True)
                aligned_count = torch.zeros_like(self.types_list, dtype=torch.float32)
                comparison = torch.eq(self.types_list.unsqueeze(1), unique_type)
                aligned_count[[index.item() for index in torch.nonzero(comparison)[:, 1]]] += count
                x.append(aligned_count)
                y.append(target)
            
            x_tensor = torch.stack(tuple(x))
            y_tensor = torch.stack(tuple(y)).reshape(-1, 1)
            xTx = torch.matmul(x_tensor.T, x_tensor)
            xTx_inv = torch.linalg.pinv(xTx)
            xTx_invx = torch.matmul(xTx_inv, x_tensor.T)
            w = torch.matmul(xTx_invx, y_tensor)

            for i, e in enumerate(unique_type):
                self.dress[str(e)](w[i])
                # print(f"{e} current: {w[i]}; mean: {self.dress[str(e)].mean}; std: {self.dress[str(e)].stddev}")
            yield batch
    
    def __len__(self):
        return len(self.dp)