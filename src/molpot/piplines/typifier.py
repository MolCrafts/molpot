import torch

from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from molpot import alias

class Typifier(IterDataPipe):
    pass

@functional_datapipe("identity_typify")
class Identity(Typifier):
    
    def __init__(self, source_dp: IterDataPipe, src: str = alias.Z, dst: str = alias.atype):
        self.dp = source_dp
        self.src = src
        self.dst = dst
        self.src2dst = {}  # index: type

    def __iter__(self):
        for d in self.dp:
            src = d[self.src]
            dst = []
            for src_value in src:
                if src_value not in self.src2dst:
                    typeid = len(self.src2dst)
                    self.src2dst[src_value] = typeid  # append
                else:
                    typeid = self.src2dst[src_value]
                dst.append(typeid)
            d[self.dst] = torch.tensor(dst, dtype=torch.long)

            yield d

    def __len__(self):
        return len(self.dp)
                