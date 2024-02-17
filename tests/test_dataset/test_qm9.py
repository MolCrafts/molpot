import pytest
import numpy as np
import molpot as mpot
from pathlib import Path

def test_load_qm9():

    dataset = mpot.dataset.QM9(None, 10, 5, False, False)
    dp = dataset.prepare()
    dl = mpot.DataLoader(dp)
    for d in dl:
        print(d)
        break

def test_load_rmd17():
    dataset = mpot.dataset.rMD17(None, 10, 5)
    dp = dataset.prepare()
    dl = mpot.DataLoader(dp)
    for d in dl:
        print(d)
        break