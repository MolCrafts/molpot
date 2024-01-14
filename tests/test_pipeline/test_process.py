import pytest
import numpy as np
import molpy as mp
from molpot import Alias
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2 import DataLoader2Iterator, DataLoader2
import molpot as mpot

alias = mpot.alias

class TestProcess:

    def test_dressing(self):

        qm9_dataset = mpot.QM9(data_dir="data/qm9", total=1000, batch_size=64)
        dp = qm9_dataset.prepare()
        # dp = dp.atomic_dress(6, alias.Z, alias.QM9.U)
        for i in dp:
            pass