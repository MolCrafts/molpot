import pytest
import torch
import molpot.potentials.nnp as nnp

class TestPiNet:

    def test_fflayer(self):
        
        i = 16
        alpha = 16
        h1, beta = 64, 64

        fflayer = nnp.FFLayer(alpha, [h1, beta])
        x = torch.randn(i, alpha)
        y = fflayer(x)
        assert y.shape == (i, beta)

        j = 8
        x = torch.randn(i, j, alpha)
        y = fflayer(x)
        assert y.shape == (i, j, beta)

    def test_pilayer(self):

        i = 16
        j = 32
        alpha = 1
        npairs = i * j
        idx_i = torch.randint(0, i, (npairs, ))
        idx_j = torch.randint(0, i, (npairs, ))
        n_basis = 4
        beta = 64
        n_basis = 4

        pilayer = nnp.PILayer(alpha, [beta], n_basis)
        x = torch.randn(i, alpha)
        center = None
        gamma = 3.0
        rc = 5.0
        cutoff_type = "f1"
        basis_fn = nnp.GaussianBasis(center, gamma, rc, n_basis)
        cutoff_fn = nnp.CutoffFunc(rc, cutoff_type)
        dist = torch.randn(npairs)
        fc = cutoff_fn(dist)
        basis = basis_fn(dist, fc=fc)
        y = pilayer(x, idx_i, idx_j, basis)
        assert y.shape == (i, beta)