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
            H2O = mpot.Frame({
                'atoms': {
                    'Z': [1, 8, 1],
                    'R': torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=Config.ftype),
                }, 
                'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                'pbc': [False, False, False],
                'pairs': {
                    'i': [0, 0, 1],
                    'j': [1, 2, 2],
                    'dist': torch.tensor([1., 1, 1], dtype=Config.ftype),
                    'diff': torch.tensor([[0, 0, 1.], [0, 1, 0], [1, 0, 0]], dtype=Config.ftype),
                },
                'bonds': {
                    'i': [0, 1],
                    'j': [1, 2],
                    'dist': torch.tensor([1., 1], dtype=Config.ftype),
                    'diff': torch.tensor([[0., 0, 1], [0, 1, 0], [1, 0, 0]], dtype=Config.ftype),
                },
            })
            frames.append(H2O)

        return frames
    return _generator

@pytest.fixture
def gen_heterogenous_frames():

    def _generator(n_frames):

        H2O =  mpot.Frame({
                'atoms': {
                    'Z': [1, 8, 1],
                    'R': [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                }, 
                'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                'pbc': [False, False, False],
                'pairs': {
                    'i': [0, 0, 1],
                    'j': [1, 2, 2],
                    'dist': [1., 1, 1],
                    'diff': [[0., 0, 1], [0, 1, 0], [1, 0, 0]],
                },
                'bonds': {
                    'i': [0, 1],
                    'j': [1, 2],
                    'dist': [1., 1],
                    'diff': [[0., 0, 1], [0, 1, 0], [1, 0, 0]],
                },
            })
        CH4 = mpot.Frame({
                'atoms': {
                    'Z': [6, 1, 1, 1, 1],
                    'R': [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]],
                }, 
                'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                'pbc': [False, False, False],
                'pairs': {
                    'i': [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4],
                    'j': [1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 5, 5],
                    'dist': [1., 1, 1],
                    'diff': [[0., 0, 1], [0, 1, 0], [1, 0, 0]],
                },
                'bonds': {
                    'i': [0, 0, 0, 0],
                    'j': [1, 2, 3, 4],
                    'dist': [1., 1],
                    'diff': [[0., 0, 1], [0, 1, 0], [1, 0, 0]],
                },
            })
        return [H2O, CH4] * int(n_frames/2)
    
    return _generator