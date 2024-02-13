import torch
import numpy.testing as npt

def create_rotation_matrix(roll, pitch, yaw):
    """Create a random rotation matrix.
    """

    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)]
    ])
    Ry = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)]
    ])
    Rz = torch.tensor([
        [torch.cos(yaw), -torch.sin(yaw), 0],
        [torch.sin(yaw), torch.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def rotate(x, roll, pitch, yaw):
    rot = create_rotation_matrix(roll, pitch, yaw)
    return torch.einsum('ixa,xy->iya', x, rot)

def assert_eqvar(layer, input, verbose=False, *args, **kwargs):
    """Assert that the variance of the output of a layer is the same as the input.
    """
    roll, pitch, yaw = torch.rand(3)
    expected = rotate(layer(input, *args, **kwargs), roll, pitch, yaw)
    actual = layer(rotate(input, roll, pitch, yaw), *args, **kwargs)
    if verbose:
        npt.assert_allclose(
            expected.detach().numpy(),
            actual.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )
    else:
        torch.allclose(
        expected,
        actual,
        rtol=1e-5,
        atol=1e-5
    )

def assert_invar(layer, input, verbose=False, *args, **kwargs):
    """Assert that the output of a layer is invariant to rotation.
    """
    roll, pitch, yaw = torch.rand(3)
    expected = layer(input, *args, **kwargs)
    actual = layer(rotate(input, roll, pitch, yaw), *args, **kwargs)
    if verbose:
        npt.assert_allclose(
            expected.detach().numpy(),
            actual.detach().numpy(),
            rtol=1e-5,
            atol=1e-5
        )
    else:
        torch.allclose(
        expected,
        actual,
        rtol=1e-5,
        atol=1e-5
    )