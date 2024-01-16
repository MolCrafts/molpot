import torch
from torch.profiler import profile, record_function, ProfilerActivity

class Profiler:

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def profile(self):
        data = next(iter(self.dataloader))
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.model(data)
        return prof