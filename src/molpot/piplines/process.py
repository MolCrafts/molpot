# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-30
# version: 0.0.1

from collections import defaultdict, deque
from typing import Sequence

import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from molpot import Config
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
    def __init__(self, source_dp: IterDataPipe, types_list: list[int], key, prop, buffer:int|None = None, ref: Sequence[float] | None = None):
        self.dp = source_dp
        self.key = key
        self.prop = prop
        self.buffer = buffer
        self.types_list = torch.tensor(types_list).to(Config.device)
        self.ref = torch.tensor(ref).to(Config.device) if ref else None

        # self.trackers = defaultdict(Tracker)

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
                
            x_tensor = torch.stack(tuple(x)).to(Config.device).to(Config.ftype)
            if self.ref:  # ref -> w
                w = self.ref
            else:
                w, residue = atomic_dress(x_tensor, torch.stack(tuple(y)).to(Config.device))
                # print(f"pool: {len(x)}, residue: {residue}, w: {w.flatten()}")
                # for atype, _w in zip(self.types_list, w):
                #     self.trackers[atype](_w)
                #     print(f"atom: {atype}, w: {_w}, mean: {self.trackers[atype].mean}, stddev: {self.trackers[atype].stddev}")
                
            for sample in batch:
                apply_dress(sample, self.types_list, self.key, self.prop, w)
            
            yield batch
    
    def __len__(self):
        return len(self.dp)
    
def atomic_dress(x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
    xTx = torch.matmul(x.T, x)
    xTx_inv = torch.linalg.pinv(xTx)
    xTx_invxT = torch.matmul(xTx_inv, x.T)
    w = torch.matmul(xTx_invxT, y)
    predict = x @ w
    residue = torch.mean((y - predict)**2)
    return w, residue

def apply_dress(frame, type_list, key, target, w):

    x = torch.eq(frame[key], type_list.unsqueeze(1)).sum(dim=-1).to(Config.device).to(Config.ftype)
    predict = x @ w
    delta = frame[target] - predict
    frame[target] = delta
    return frame