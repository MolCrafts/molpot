import pytest
import torch

import molpot as mp


def test_wigner_3j_symmetry() -> None:
    assert torch.allclose(mp.irrep.wigner_3j(1, 2, 3), mp.irrep.wigner_3j(1, 3, 2).transpose(1, 2))
    assert torch.allclose(mp.irrep.wigner_3j(1, 2, 3), mp.irrep.wigner_3j(2, 1, 3).transpose(0, 1))
    assert torch.allclose(mp.irrep.wigner_3j(1, 2, 3), mp.irrep.wigner_3j(3, 2, 1).transpose(0, 2))
    assert torch.allclose(mp.irrep.wigner_3j(1, 2, 3), mp.irrep.wigner_3j(3, 1, 2).transpose(0, 1).transpose(1, 2))
    assert torch.allclose(mp.irrep.wigner_3j(1, 2, 3), mp.irrep.wigner_3j(2, 3, 1).transpose(0, 2).transpose(1, 2))


@pytest.mark.parametrize("l1,l2,l3", [(1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 2, 2)])
def test_wigner_3j(l1, l2, l3) -> None:
    abc = mp.irrep.rand_angles(10)

    C = mp.irrep.wigner_3j(l1, l2, l3)
    D1 = mp.Irrep(l1, 1).D_from_angles(*abc)
    D2 = mp.Irrep(l2, 1).D_from_angles(*abc)
    D3 = mp.Irrep(l3, 1).D_from_angles(*abc)

    C2 = torch.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
    # assert (C - C2).abs().max() < float_tolerance
    assert torch.allclose(C, C2, atol=1e-3)


def test_cartesian() -> None:
    abc = mp.irrep.rand_angles(10)
    R = mp.irrep.angles_to_matrix(*abc)
    D = mp.irrep.wigner_D(1, *abc)
    # assert (R - D).abs().max() < float_tolerance
    torch.allclose(R, D, atol=1e-3)


def commutator(A, B):
    return A @ B - B @ A


@pytest.mark.parametrize("j", [0, 1 / 2, 1, 3 / 2, 2, 5 / 2])
def test_su2_algebra(j) -> None:
    X = mp.irrep.su2_generators(j)
    assert torch.allclose(commutator(X[0], X[1]), X[2], atol=1e-3)
    assert torch.allclose(commutator(X[1], X[2]), X[0], atol=1e-3)