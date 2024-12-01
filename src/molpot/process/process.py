import torch
from collections import defaultdict, deque
from .base import Process
from molpot import Config

class Normalizer(Process):

    def __init__(self, keys: list):
        self.keys = keys
        self.result = defaultdict(lambda: (0, 0., 0.))

    def __call__(self, tensordict):

        for key in self.keys:
            new_value = tensordict[key]
            count, mean, m2 = self.result[key]
            count += 1
            delta = new_value - mean
            mean += delta / count
            delta2 = new_value - mean
            m2 += delta * delta2
            self.result[key] = (count, mean, m2)
            # TODO: how to access?
            if count < 2:
                (mean_, var, sample_var) = (new_value, 0., 0.)
            else:
                (mean_, var, sample_var) = (mean, m2 / count, m2 / (count - 1))

        return tensordict


class AtomicDressing(Process):
    """
    http://mlwiki.org/index.php/Normal_Equation
    """
    def __init__(self, key, ref = None, buffer: int | None = None):
        self.key = key
        # self.prop = prop
        self.buffer = buffer
        # self.types_list = torch.tensor(types_list).to(Config.device)
        self.ref = torch.tensor(ref).to(Config.device) if ref else None

        # self.trackers = defaultdict(Tracker)

    def __call__(self, tensordict):

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
