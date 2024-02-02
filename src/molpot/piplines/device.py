from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe

@functional_datapipe("to_device")
class Normalizer(IterDataPipe):

    def __init__(self, source_dp: IterDataPipe, device):
        self.dp = source_dp
        self.device = device

    def __iter__(self):
        for d in self.dp:
            yield [{k:v.to(self.device) for k, v in x.items()} for x in d]

    def __len__(self):
        return len(self.dp)