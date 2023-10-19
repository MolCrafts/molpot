import torch


def get_optimizer(config):
    if config["type"] == "adam":
        optimizer = torch.optim.Adam(config["args"])
    else:
        raise NotImplementedError

    if "lr_scheduler" in config:
        lr_scheduler = getattr(torch.optim.lr_scheduler, config["lr_scheduler"]["type"])
        opt = lr_scheduler(optimizer, **config["lr_scheduler"]["args"])

    return opt


def get_lr_scheduler(config):
    if config["type"] == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(config["args"])
    else:
        raise NotImplementedError

    return lr_scheduler
