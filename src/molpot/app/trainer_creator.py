import argparse
import collections
import torch
import numpy as np
from .config_parser import ConfigParser
from ..piplines import get_data_loader
from ..potentials import get_potential
from .utils import prepare_device
from ..trainer import get_criterion, Trainer, get_metric, get_optimizer, get_lr_scheduler


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def create_trainer(config_parser: ConfigParser):
    config = config_parser.config
    data_loader = get_data_loader(config["data_loader"]).prepare()
    if isinstance(data_loader, tuple):
        data_loader, valid_data_loader = data_loader
    model = get_potential(config["model"])
    # TODO: convert to functional

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = get_criterion(config["loss"])
    metrics = [get_metric(met) for met in config["metrics"]]

    # convert nn.Modules to functional
    # params = dict(model.named_parameters())
    # f = torch.func.functional_call(model, params)
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(config["optimizer"])
    lr_scheduler = get_lr_scheduler(config["lr_scheduler"])
    optimizer = optimizer(model.parameters())
    lr_scheduler = lr_scheduler(optimizer)

    return Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    # trainer.train()
