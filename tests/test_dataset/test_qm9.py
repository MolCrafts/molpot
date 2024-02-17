import pytest
import numpy as np
import molpot as mpot
from pathlib import Path

def test_load_qm9_in_memory():

    dataset = mpot.QM9(None, 10, 5, False, False)
    dp = dataset.prepare()
    dl = mpot.DataLoader(dp)
    for d in dl:
        print(d)
        assert False