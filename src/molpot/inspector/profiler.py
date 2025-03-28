import torch
from torch.profiler import profile, record_function, ProfilerActivity

class Profiler:

    def __init__(self, model, dataloader):

        self.model = model
        self.dataloader = dataloader

    def profile(self):
        data = next(iter(self.dataloader))
        # for key in data:
        #     data[key] = data[key].to(self.model.device)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                self.model(data)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return prof