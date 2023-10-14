import pytest
import numpy as np
import molpy as mp
from molpot import kw
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2 import DataLoader2Iterator, DataLoader2

class TestCollateFrame:

    def test_aligned(self):

        frames = IterableWrapper([
            mp.Frame(
                **{
                    kw.natoms: 2,
                    kw.xyz: np.array([[0, 0, 0], [1, 0, 1]]),
                    kw.Z: [0, 1],
                }
            ),
            mp.Frame(
                **{
                    kw.natoms: 2,
                    kw.xyz: np.array([[2, 0, 0], [3, 0, 1]]),
                    kw.Z: [2, 3],
                }
            ),
        ])
        dp = frames.batch(batch_size=2).collate_frames()
        dl = DataLoader2(dp)
        for batch in dl:
            print(batch)
