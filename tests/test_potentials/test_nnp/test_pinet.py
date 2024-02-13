import pytest
import torch
import molpot.potentials.nnp as nnp
from molpot import Config
from .utils import assert_eqvar, assert_invar


class TestPiNet:

    def test_pilayer(self):

        n_atoms = 2
        n_pairs = 2
        n_props = 1
        n_channels = 1
        n_hidden = [3, 4]
        n_basis = 6

        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        idx_j = torch.randint(0, n_atoms, (n_pairs,))
        input = torch.rand(n_atoms, n_props, n_channels)
        basis = torch.rand(n_pairs, n_basis)

        layer = nnp.pinet.PILayer(n_channels, n_hidden, n_basis)
        output = layer(input, idx_i, idx_j, basis)
        assert output.shape == (n_pairs, n_props, n_hidden[-1])
        assert_invar(layer, input, True, idx_i, idx_j, basis)

    def test_pixplayer(self):

        nsamples = 1
        ndims = 3
        nchannels = 1
        nnbors = 1
        px = torch.rand((nsamples, ndims, nchannels))
        idx_i = torch.randint(0, nsamples, (nnbors,))
        idx_j = torch.randint(0, nsamples, (nnbors,))

        pix = nnp.pinet.PIXLayer()
        out = pix(px, idx_i, idx_j)
        assert out.shape == (nnbors, ndims, nchannels)
        assert_eqvar(pix, px, True, idx_i, idx_j)