import pytest
import torch

from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF

class TestBasisFunction:

    def test_gaussian(self):

        n_atoms = 10
        n_dim = 3
        n_channels = 5
        n_basis = 16
        cutoff = 5.0

        input = torch.rand(n_atoms)
        cutoff_fn = CosineCutoff(cutoff)
        layer = GaussianRBF(n_basis, cutoff, 0, False)
        output = layer(input, cutoff_fn(input))
        assert output.shape == (n_atoms, n_basis)

class TestCutoffFunction:

    def test_cosine(self):

        n_atoms = 10
        cutoff = 5.0

        input = torch.rand(n_atoms)
        layer = CosineCutoff(cutoff)
        output = layer(input)
        assert layer.cutoff == cutoff
        assert output.shape == (n_atoms, )
