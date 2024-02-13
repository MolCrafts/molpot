from typing import Any
import torch
import numpy.testing as npt
from functools import partial


def create_rotation_matrix(roll, pitch, yaw):
    """Create a random rotation matrix."""

    Rx = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)],
        ]
    )
    Ry = torch.tensor(
        [
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0, torch.cos(pitch)],
        ]
    )
    Rz = torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    return Rz @ Ry @ Rx


def rotate(x, roll, pitch, yaw):
    ndim = x.ndim - 2
    rot = create_rotation_matrix(roll, pitch, yaw)
    if ndim == 0:
        return torch.einsum("xc, xy->yc", x, rot)
    elif ndim == 1:
        return torch.einsum("ixc, xy->iyc", x, rot)
    elif ndim == 2:
        return torch.einsum("xw, ixyc, yz->iwzc", rot, x, rot)


def foreach_rotate(x, eqvar_index, roll, pitch, yaw):
    if isinstance(x, (list, tuple)):
        return [
        rotate(x, roll, pitch, yaw) if i in eqvar_index else x for i, x in enumerate(x)
    ]
    elif isinstance(x, dict):
        return {
            k: rotate(v, roll, pitch, yaw) if k in eqvar_index else v for k, v in x.items()
        }


def allclose(a, b, verbose: bool = True, rtol=1e-5, atol=1e-5):
    if verbose:
        npt.assert_allclose(
            a.detach().numpy(), b.detach().numpy(), rtol=rtol, atol=atol
        )
    else:
        torch.allclose(a, b, rtol=rtol, atol=atol)


def assert_eqvar(
    layer,
    input: list[Any],
    eqvar_index: list[int],
    verbose: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Assert that the variance of the output of a layer is the same as the input."""
    roll, pitch, yaw = torch.rand(3)
    eqvars = foreach_rotate(input, eqvar_index, roll, pitch, yaw)
    expected = rotate(layer(*input), roll, pitch, yaw)
    actual = layer(*eqvars)

    allclose(expected, actual, verbose, rtol, atol)


def assert_invar(
    layer,
    input: list[Any],
    eqvar_index: list[int],
    verbose: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """Assert that the output of a layer is invariant to rotation."""
    roll, pitch, yaw = torch.rand(3)
    eqvars = foreach_rotate(input, eqvar_index, roll, pitch, yaw)
    expected = layer(*input)
    actual = layer(*eqvars)

    allclose(expected, actual, verbose, rtol, atol)
