import pytest
import numpy as np
import molpot as mpot
from pathlib import Path


def test_load_qm9():

    dataset = mpot.dataset.QM9(
        save_dir=None,
        total=10,
        batch_size=5,
        atom_ref=False,
        remove_uncharacterized=False,
    )
    dl = mpot.DataLoader(dataset, batch_size=1)
    for d in dl:
        print(d)
        break
