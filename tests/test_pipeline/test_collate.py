import molpy as mp
import numpy as np
import pytest
import torch
from torchdata.dataloader2 import DataLoader2, DataLoader2Iterator
from torchdata.datapipes.iter import IterableWrapper

from molpot import Alias


class TestCollateFrame:

    # def test_aligned(self):

    #     frames = IterableWrapper([
    #         mp.Frame(
    #             **{
    #                 Alias.n_atoms: 2,
    #                 Alias.xyz: np.array([[0, 0, 0], [1, 0, 1]]),
    #                 Alias.Z: [0, 1],
    #             }
    #         ),
    #         mp.Frame(
    #             **{
    #                 Alias.n_atoms: 2,
    #                 Alias.xyz: np.array([[2, 0, 0], [3, 0, 1]]),
    #                 Alias.Z: [2, 3],
    #             }
    #         ),
    #     ])
    #     dp = frames.batch(batch_size=2).collate_frames()
    #     dl = DataLoader2(dp)
    #     for batch in dl:
    #         print(batch)

    def test_dict(self):

        frame = IterableWrapper([
            {
                Alias.n_atoms: torch.tensor([2]),
                Alias.xyz: torch.tensor([[0, 0, 0], [1, 0, 1]]),
                Alias.Z: torch.tensor([0, 1]),
            },
            {
                Alias.n_atoms: torch.tensor([2]),
                Alias.xyz: torch.tensor([[2, 0, 0], [3, 0, 1]]),
                Alias.Z: torch.tensor([2, 3]),
            },
            {
                Alias.n_atoms: torch.tensor([3]),
                Alias.xyz: torch.tensor([[0, 0, 0], [1, 0, 1], [2, 0, 0]]),
                Alias.Z: torch.tensor([0, 1, 2]),
            }
        ])
        dp = frame.batch(batch_size=3).collate_data()
        dl = DataLoader2(dp)
        batch = next(iter(dl))
        assert torch.equal(batch[Alias.idx_m], torch.tensor([0, 0, 1, 1, 2, 2, 2]))
        assert torch.equal(batch[Alias.n_atoms], torch.tensor([2, 2, 3]))
        assert batch[Alias.xyz].shape == (7, 3)
        assert torch.equal(batch[Alias.Z], torch.tensor([0, 1, 2, 3, 0, 1, 2]))