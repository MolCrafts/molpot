import subprocess
from pathlib import Path

import pytest
import torch

import molpot as mpot
from molpot import alias, Config


@pytest.fixture
def gen_homogenous_frames():

    def _generator(n_frame):
        frames = []
        for i in range(n_frame):
            R = torch.tensor(
                            [
                                [0.00000, -0.06556, 0.00000],
                                [0.75695, 0.52032, 0.00000],
                                [-0.75695, 0.52032, 0.00000],
                            ],
                            dtype=Config.ftype,
                            requires_grad=True,
                        )
            pair_i = [0, 0, 1]
            pair_j = [1, 2, 2]
            pair_diff = R[pair_j] - R[pair_i]
            pair_dist = torch.linalg.norm(pair_diff, dim=-1)
            H2O = mpot.Frame(
                {
                    "atoms": {
                        "Z": [8, 1, 1],
                        "type": [1, 0, 0],
                        "R": R
                    },
                    "cell": None,
                    "pbc": [False, False, False],
                    "pairs": {
                        "i": pair_i,
                        "j": pair_j,
                        "dist": pair_dist,
                        "diff": pair_diff,
                    },
                    "bonds": {
                        "i": [0, 0],
                        "j": [1, 2],
                        "dist": torch.tensor(
                            [0.9573, 0.9573], dtype=Config.ftype, requires_grad=True
                        ),
                        "diff": torch.tensor(
                            [[0.7570, 0.5859, 0.0000], [-0.7570, 0.5859, 0.0000]],
                            dtype=Config.ftype,
                            requires_grad=True,
                        ),
                    },
                }
            )
            frames.append(H2O)

        return frames

    return _generator


@pytest.fixture
def gen_heterogenous_frames():

    def _generator(n_frames):

        R = torch.tensor(
            [[0., 0, 0], [0, 0, 1], [0, 1, 0]]
        )
        pair_i = [0, 0, 1]
        pair_j = [1, 2, 2]
        pair_diff = R[pair_j] - R[pair_i]
        pair_dist = torch.linalg.norm(pair_diff, dim=-1)

        H2O = mpot.Frame(
            {
                "atoms": {
                    "Z": [1, 8, 1],
                    "R": R,
                },
                "cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                "pbc": [False, False, False],
                "pairs": {
                    "i": pair_i,
                    "j": pair_j,
                    "dist": pair_dist,
                    "diff": pair_diff,
                },
                "bonds": {
                    "i": [0, 1],
                    "j": [1, 2],
                    "dist": [1.0, 1],
                    "diff": [[0.0, 0, 1], [0, 1, 0], [1, 0, 0]],
                },
            }
        )

        R = torch.tensor(
            [[0., 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]]
        )
        pair_i = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
        pair_j = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
        pair_diff = R[pair_j] - R[pair_i]
        pair_dist = torch.linalg.norm(pair_diff, dim=-1)        

        CH4 = mpot.Frame(
            {
                "atoms": {
                    "Z": [6, 1, 1, 1, 1],
                    "R": R,
                },
                "cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                "pbc": [False, False, False],
                "pairs": {
                    "i": pair_i,
                    "j": pair_j,
                    "dist": pair_dist,
                    "diff": pair_diff,
                },
                "bonds": {
                    "i": [0, 0, 0, 0],
                    "j": [1, 2, 3, 4],
                    "dist": [1.0, 1],
                    "diff": [[0.0, 0, 1], [0, 1, 0], [1, 0, 0]],
                },
            }
        )
        return [H2O, CH4] * int(n_frames / 2)

    return _generator
