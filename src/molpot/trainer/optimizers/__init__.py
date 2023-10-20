from functools import partial
import torch


def get_optimizer(config):
    if config["type"] == "Adam":
        optimizer = partial(torch.optim.Adam, **config["args"])
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(config):
    if config["type"] == "StepLR":
        lr_scheduler = partial(torch.optim.lr_scheduler.StepLR, **config["args"])
    else:
        raise NotImplementedError

    return lr_scheduler
