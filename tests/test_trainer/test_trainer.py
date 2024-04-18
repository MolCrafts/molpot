# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-03
# version: 0.0.1

import torch
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from molpot.pipline.dataloaders import create_dataloader
from molpot.potential.classical.pair.lj import LJ126
from molpot.trainer import Trainer


def test_trainer():

    