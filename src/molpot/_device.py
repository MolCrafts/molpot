import torch

def set_device(device_type: str):
    """
    Setup GPU device if available, move model into configured device
    """
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

device = torch.device("cuda:0")