import pytest
import torch

from molpot.potential.nnp.radial import GaussianRBF
from molpot.potential.nnp.cutoff import CosineCutoff


class TestBasisFunction:

    def test_gaussian_rbf(self):

        n_pairs = 10
        n_basis = 16
        cutoff = 5.0

        d_ij = torch.rand(n_pairs)
        cutoff_fn = CosineCutoff(cutoff)
        layer = GaussianRBF(n_basis, cutoff, 0, False)
        output = layer(d_ij, cutoff_fn(d_ij))
        assert output.shape == (n_pairs, n_basis)

class TestCutoffFunction:

    def test_cosine(self):

        n_atoms = 10
        cutoff = 5.0

        input = torch.rand(n_atoms)
        layer = CosineCutoff(cutoff)
        output = layer(input)
        assert layer.cutoff == cutoff
        assert output.shape == (n_atoms, )
