import torch

class Config:

    device: torch.device = torch.device("cpu")
    stype: torch.dtype = torch.int32
    ftype: torch.dtype = torch.float32 

    @classmethod
    def set_device(cls, device_type:str):
        if device_type == "cpu":
            device = torch.device("cpu")
        elif device_type == "gpu" or device_type == "cuda":
            n_gpu = torch.cuda.device_count()
            if n_gpu == 0:
                print("Warning: There\'s no GPU available on this machine,"
                    "training will be performed on CPU.")
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:0")
        return device