import pytest
import torch
from molpot import Config
from functools import partial
import numpy.testing as npt

cos = torch.cos
sin = torch.sin


def create_random_rot_mat():

    roll = torch.rand(1)
    yaw = torch.rand(1)
    pitch = torch.rand(1)

    RX = torch.tensor(
        [[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]]
    )

    RY = torch.tensor(
        [[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]]
    )

    RZ = torch.tensor([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])

    R = RZ @ RY @ RX

    return R


def rotate(x):
    rot = create_random_rot_mat()
    return torch.einsum("ix,xy->iy", x, rot)


def assert_invar(layer, input, *args, **kwargs):

    assert npt.assert_allclose(
        layer(rotate(input), *args, **kwargs).detach().numpy(),
        layer(input, *args, **kwargs).detach().numpy(),
        rtol=1e-1,
        atol=1e-1,
    )


def assert_eqvar(layer, input, *args, **kwargs):

    assert torch.allclose(
        layer(rotate(input), *args, **kwargs).detach().numpy(),
        rotate(layer(input, *args, **kwargs)).detach().numpy()
    )


class TestPiNet:

    def test_PILayer(self):
        from molpot.potentials.nnp.pinet import PILayer

        n_atoms = 10
        n_prop = 5
        n_basis = 6
        n_pairs = 12

        layer = PILayer(n_prop, [2, 3, 4], n_basis)
        input = torch.rand(n_atoms, n_prop)

        assert_invar(
            layer,
            input,
            idx_i=torch.randint(0, n_atoms, (n_pairs,), dtype=Config.stype),
            idx_j=torch.randint(0, n_atoms, (n_pairs,), dtype=Config.stype),
            basis=torch.randn(n_pairs, n_basis, dtype=Config.ftype),
        )
