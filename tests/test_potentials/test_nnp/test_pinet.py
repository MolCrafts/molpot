import pytest
import torch
import molpot.potentials.nnp as nnp
from molpot import Config, Alias
from molpot.potentials.nnp.pinet import DotLayer
from .utils import assert_eqvar, assert_invar, rotate


class TestPiNet:

    def test_pilayer(self):

        n_atoms = 10
        n_pairs = 2
        n_dim = 3
        n_channels = 16
        n_hidden = [3, 4]
        n_basis = 16

        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        idx_j = torch.randint(0, n_atoms, (n_pairs,))
        basis = torch.rand(n_pairs, n_basis)
        layer = nnp.pinet.PILayer(n_channels, n_hidden, n_basis)

        # p1
        input = torch.rand(n_atoms, n_channels)
        output = layer(input, idx_i, idx_j, basis)
        assert output.shape == (n_pairs, n_hidden[-1])

        # p3
        input = torch.rand(n_atoms, n_dim, n_channels)
        output = layer(input, idx_i, idx_j, basis)
        assert output.shape == (n_pairs, n_dim, n_hidden[-1])
        assert_eqvar(layer, [input, idx_i, idx_j, basis], [0])

        # p5
        input = torch.rand(n_atoms, n_dim, n_dim, n_channels)
        output = layer(input, idx_i, idx_j, basis)
        assert output.shape == (n_pairs, n_dim, n_dim, n_hidden[-1])
        assert_eqvar(layer, [input, idx_i, idx_j, basis], [0])


    def test_mlp_layer(self):

        n_atoms = 5
        n_channels = 6
        n_dim = 3
        n_neuraons = [6, 4]
        px = torch.rand(n_atoms, n_dim, n_channels)

        layer = nnp.layers.build_mlp(n_neuraons)
        out = layer(px)
        assert out.shape == (n_atoms, n_dim, n_neuraons[-1])
        assert_eqvar(layer, [px], [0])

    def test_dotlayer(self):

        n_atoms = 10
        n_dim = 3
        n_channels = 5

        # p3
        input = torch.rand(n_atoms, n_dim, n_channels)
        layer = DotLayer()
        output = layer(input)
        assert output.shape == (n_atoms, n_channels)
        assert_invar(layer, [input], [0])

        # p5
        input = torch.rand(n_atoms, n_dim, n_dim, n_channels)
        output = layer(input)
        assert output.shape == (n_atoms,  n_channels)
        assert_invar(layer, [input], [0])

    def test_scalelayer(self):

        n_atoms = 10
        n_dim = 3
        n_channels = 5
        p1 = torch.rand(n_atoms, n_channels)
        layer = nnp.pinet.ScaleLayer()

        # p3
        p3 = torch.rand(n_atoms, n_dim, n_channels)
        output = layer(p3, p1)
        assert output.shape == (n_atoms, n_dim, n_channels)
        assert_eqvar(layer, [p3, p1], [0])

        # p5
        p5 = torch.rand(n_atoms, n_dim, n_dim, n_channels)
        output = layer(p5, p1)
        assert output.shape == (n_atoms, n_dim, n_dim, n_channels)
        assert_eqvar(layer, [p5, p1], [0])

    def test_iplayer(self):

        n_atoms = 10
        n_pairs = 12
        n_dim = 3
        n_channels = 5
        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        i1 = torch.rand(n_pairs, n_channels)
        i3 = torch.rand(n_pairs, n_dim, n_channels)
        i5 = torch.rand(n_pairs, n_dim, n_dim, n_channels)
        p1 = torch.rand(n_atoms, n_channels)
        p3 = torch.rand(n_atoms, n_dim, n_channels)
        p5 = torch.rand(n_atoms, n_dim, n_dim, n_channels)


        layer = nnp.pinet.IPLayer()
        output = layer(i1, idx_i, p1)
        assert output.shape == (n_atoms, n_channels)

        output = layer(i3, idx_i, p3)
        assert output.shape == (n_atoms, n_dim, n_channels)
        assert_eqvar(layer, [i3, idx_i, p3], [0])

        output = layer(i5, idx_i, p5)
        assert output.shape == (n_atoms, n_dim, n_dim, n_channels)
        assert_eqvar(layer, [i5, idx_i, p5], [0])

    def test_addlayer(self):

        n_dim = 3
        n_pairs = 12
        n_channels = 5
        i3 = torch.rand(n_pairs, n_dim, n_channels)
        # p3
        layer = nnp.pinet.AddLayer(3)
        output = layer(i3, i3)
        assert output.shape == (n_pairs, n_dim, n_channels)
        assert_eqvar(layer, [i3, i3], [0, 1])
        # p5
        layer = nnp.pinet.AddLayer(5)
        i5 = torch.rand(n_pairs, n_dim, n_dim, n_channels)
        output = layer(i5, i3)
        assert output.shape == (n_pairs, n_dim, n_dim, n_channels)
        assert_eqvar(layer, [i5, i3], [0, 1])
        

    def test_gcblock_p1(self):

        n_atoms = 10
        n_pairs = 12
        n_channels = 16
        n_basis = 6
        pp_nodes = [16, 16]
        pi_nodes = [16, 16]
        ii_nodes = [16, 16]
        p1 = torch.rand(n_atoms, n_channels)
        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        idx_j = torch.randint(0, n_atoms, (n_pairs,))
        basis = torch.rand(n_pairs, n_basis)
        activation = torch.nn.ReLU()

        layer = nnp.pinet.GCBlockP1(pp_nodes, pi_nodes, ii_nodes, n_basis, activation)
        output = layer(p1, idx_i, idx_j, basis)
        assert output.shape == (n_atoms, n_channels)

    def test_pinet_p1(self):

        n_atoms = 10
        n_atomtypes = 5
        n_pairs = 12
        n_basis = 16
        depth = 4
        cutoff = 1.0
        cutoff_fn = nnp.layers.CosineCutoff(1.0)
        radial_basis = nnp.layers.GaussianRBF(n_basis, cutoff)
        pp_nodes = [16, 16]
        pi_nodes = [16, 16]
        ii_nodes = [16, 16]
        p1 = torch.randint(0, n_atomtypes, (n_atoms,))
        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        idx_j = torch.randint(0, n_atoms, (n_pairs,))

        layer = nnp.pinet.PiNet(n_basis, depth, radial_basis, cutoff_fn, pp_nodes, pi_nodes, ii_nodes)
        output = layer({
            Alias.Rij : torch.rand(n_pairs, 3),
            Alias.Z : p1,
            Alias.idx_i : idx_i,
            Alias.idx_j : idx_j
        })
        assert output[Alias.pinet.output_p1].shape == (n_atoms, 1)
        actual = layer({
            Alias.Rij : rotate(torch.rand(n_pairs, 3), *torch.rand(3)),
            Alias.Z : p1,
            Alias.idx_i : idx_i,
            Alias.idx_j : idx_j
        })[Alias.pinet.output_p1]
        expect = layer({
            Alias.Rij : torch.rand(n_pairs, 3),
            Alias.Z : p1,
            Alias.idx_i : idx_i,
            Alias.idx_j : idx_j
        })[Alias.pinet.output_p1]
        assert torch.allclose(actual, expect)

    def test_gcblock_p3(self):

        n_atoms = 10
        n_pairs = 12
        n_dim = 3
        n_channels = 16
        n_basis = 6
        pp_nodes = [16, 16]
        pi_nodes = [16, 16]
        ii_nodes = [16, 16]
        p1 = torch.rand(n_atoms, n_channels)
        p3 = torch.rand(n_atoms, n_dim, n_channels)
        r3 = torch.rand(n_pairs, n_dim)
        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        idx_j = torch.randint(0, n_atoms, (n_pairs,))
        basis = torch.rand(n_pairs, n_basis)
        activation = torch.nn.ReLU()

        layer = nnp.pinet.GCBlockP3(pp_nodes, pi_nodes, ii_nodes, n_basis, activation)
        output = layer(p1, p3, r3, idx_i, idx_j, basis)
        assert output[0].shape == (n_atoms, n_channels)
        assert output[1].shape == (n_atoms, n_dim, n_channels)
        assert_eqvar(layer, [p1, p3, r3, idx_i, idx_j, basis], [1, 2], [1])

    def test_pinet_p3(self):

        n_atoms = 10
        n_atomtypes = 5
        n_pairs = 12
        n_basis = 16
        depth = 4
        cutoff = 1.0
        cutoff_fn = nnp.layers.CosineCutoff(1.0)
        radial_basis = nnp.layers.GaussianRBF(n_basis, cutoff)
        pp_nodes = [16, 16]
        pi_nodes = [16, 16]
        ii_nodes = [16, 16]
        p1 = torch.randint(0, n_atomtypes, (n_atoms,))
        idx_i = torch.randint(0, n_atoms, (n_pairs,))
        idx_j = torch.randint(0, n_atoms, (n_pairs,))

        layer = nnp.pinet.PiNetP3(n_basis, depth, radial_basis, cutoff_fn, pp_nodes, pi_nodes, ii_nodes)
        output = layer({
            Alias.Rij : torch.rand(n_pairs, 3),
            Alias.Z : p1,
            Alias.idx_i : idx_i,
            Alias.idx_j : idx_j
        })
        assert output[Alias.pinet.output_p1].shape == (n_atoms, 1)
        actual = layer({
            Alias.Rij : rotate(torch.rand(n_pairs, 3), *torch.rand(3)),
            Alias.Z : p1,
            Alias.idx_i : idx_i,
            Alias.idx_j : idx_j
        })[Alias.pinet.output_p1]
        expect = layer({
            Alias.Rij : torch.rand(n_pairs, 3),
            Alias.Z : p1,
            Alias.idx_i : idx_i,
            Alias.idx_j : idx_j
        })[Alias.pinet.output_p1]
        assert torch.allclose(actual, expect)