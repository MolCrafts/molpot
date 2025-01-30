import pytest
import torch
from molpot import Config, alias
import molpot as mpot
import numpy.testing as npt

def rot3d(p3, axis, theta):
    r"""Rotate 3D coordinates by theta around axis by applying Rodrigues' rotation formula.
    Args:
        p3 (torch.Tensor): 3D coordinates of shape (n_atoms, 3, n_features).
        axis (torch.Tensor): Rotation axis of shape (3,).
        theta (torch.Tensor): Rotation angle of shape (1,) in degrees.
    """
    axis = axis / torch.norm(axis)
    theta = torch.deg2rad(theta)
    a = torch.cos(theta / 2)
    b, c, d = -axis * torch.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = torch.tensor(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ],
        dtype=Config.ftype,
    )
    return torch.einsum("ix...,xy->iy...", p3, rot)


class ModuleTester:
    def __init__(self, model):
        self.model = model

    def test_shape(self, input_data, expected_output_shape):
        output = self.model(*input_data)
        assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"

    def test_equivariance(self, input_data, transform_input, transform_output):
        original_output = transform_output(self.model(*input_data))
        transformed_input = transform_input(input_data)
        transformed_output = self.model(*transformed_input)
        assert torch.allclose(original_output, transformed_output, atol=1e-4, rtol=1e-4), "Model is not equivariant"

    def test_invariance(self, input_data, transform_fn):
        original_output = self.model(*input_data)
        transformed_input = transform_fn(input_data)
        transformed_output = self.model(*transformed_input)
        assert torch.allclose(original_output, transformed_output, atol=1e-4, rtol=1e-4), "Model is not invariant"


class TestPiNet:

    @pytest.fixture
    def config(self):
        return {
            'n_atoms': 5,
            'n_features': 32,
            'n_basis': 10,
            'n_frames': 9,
            'n_batch': 3
        }

    @pytest.fixture
    def n_atoms(self, config):
        return config['n_atoms']

    @pytest.fixture
    def n_features(self, config):
        return config['n_features']

    @pytest.fixture
    def n_basis(self, config):
        return config['n_basis']

    @pytest.fixture
    def n_frames(self, config):
        return config['n_frames']

    @pytest.fixture
    def n_batch(self, config):
        return config['n_batch']

    @pytest.fixture
    def n_pairs(self, n_atoms):
        return torch.randint(n_atoms, n_atoms**2, ())

    @pytest.fixture
    def p1(self, n_atoms, n_features):
        return torch.rand(n_atoms, 1, n_features, dtype=Config.ftype)

    @pytest.fixture
    def p3(self, n_atoms, n_features):
        return torch.rand(n_atoms, 3, n_features, dtype=Config.ftype)

    @pytest.fixture
    def pair_indices(self, n_atoms, n_pairs):
        return torch.randint(0, n_atoms, (n_pairs, 2), dtype=torch.int64)

    @pytest.fixture
    def pair_i(self, pair_indices):
        return pair_indices[:, 0]

    @pytest.fixture
    def pair_j(self, pair_indices):
        return pair_indices[:, 1]

    @pytest.fixture
    def basis(self, n_pairs, n_basis):
        return torch.rand(n_pairs, n_basis, dtype=Config.ftype)

    @pytest.fixture
    def i1(self, p1, pair_i):
        return p1[pair_i]

    @pytest.fixture
    def rotate(self, p1):
        pass

    def test_fflayer(self, n_atoms, n_features, p1):
        from molpot.potential.nnp.pinet import FFLayer

        ppl = FFLayer(n_features, n_features, n_features * 2)
        test_utils = ModuleTester(ppl)
        test_utils.test_shape((p1,), (n_atoms, 1, n_features * 2))

    def test_pilayer(self, p1, pair_i, pair_j, basis, n_pairs, n_features, n_basis):
        from molpot.potential.nnp.pinet import PILayer

        pil = PILayer(n_features, n_features * n_basis)
        test_utils = ModuleTester(pil)
        test_utils.test_shape((p1, pair_i, pair_j, basis), (n_pairs, 1, n_features))

    def test_iplayer(self, p1, n_atoms, i1, pair_i, n_features):
        from molpot.potential.nnp.pinet import IPLayer

        ipl = IPLayer()
        test_utils = ModuleTester(ipl)
        test_utils.test_shape((i1, pair_i, p1), (n_atoms, 1, n_features))

    def test_pixlayer(self, p3, pair_i, pair_j, n_features):
        from molpot.potential.nnp.pinet import PIXLayer

        pixl = PIXLayer(n_features, n_features)
        test_utils = ModuleTester(pixl)

        axis = torch.rand(3)
        theta = torch.rand(1)

        def rotate_fn(data):
            p3, pair_i, pair_j = data
            return (rot3d(p3, axis, theta=theta), pair_i, pair_j)
        
        def rotate_output(data):
            return rot3d(data, axis, theta=theta)

        test_utils.test_equivariance((p3, pair_i, pair_j), rotate_fn, rotate_output)

    def test_scalelayer(self, p3, p1):
        from molpot.potential.nnp.pinet import ScaleLayer
        scalel = ScaleLayer()
        test_utils = ModuleTester(scalel)
        test_utils.test_shape((p3, p1), p3.shape)

    def test_selfdotlayer(self, p3):
        from molpot.potential.nnp.pinet import SelfDotLayer
        sdl = SelfDotLayer()
        test_utils = ModuleTester(sdl)
        test_utils.test_shape((p3,), (p3.shape[0], 1, p3.shape[-1]))

    def test_pinet3(self, gen_homogenous_frames, n_frames, n_batch, n_basis):
        r_cutoff = 5.0
        frames = gen_homogenous_frames(n_frames)
        dataloader = mpot.DataLoader(frames, batch_size=n_batch, shuffle=False)
        pinet = mpot.potential.nnp.PiNet(
            depth=4,
            basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
            cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
        )
        readout = mpot.potential.nnp.readout.Atomwise([16, 1], in_keys=[("pinet", "p1"), alias.atom_batch], out_keys=("predict", "energy"))

        model = mpot.PotentialSeq(pinet, readout)

        test_utils = ModuleTester(model)

        axis = torch.rand(3)
        theta = torch.rand(1)

        def rotate_fn(batch):
            batch[alias.xyz] = rot3d(batch[alias.xyz], axis, theta)
            return batch
        
        def rotate_output(batch):
            return rot3d(batch["pinet", "p3"], axis, theta)

        for batch in dataloader:

            test_utils.test_equivariance((batch, ), rotate_fn, rotate_output)

    # def test_pinet1(self, gen_homogenous_frames, n_frames, n_batch, n_basis):
    #     r_cutoff = 5.0
    #     frames = gen_homogenous_frames(n_frames)
    #     dataloader = mpot.DataLoader(frames, batch_size=n_batch, shuffle=False)
    #     pinet = mpot.potential.nnp.PiNet(
    #         depth=4,
    #         basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
    #         cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
    #     )
    #     readout = mpot.potential.nnp.readout.Atomwise([16, 1], in_keys=("pinet", "p1"), out_keys=("predict", "energy"))

    #     model = mpot.PotentialSeq(pinet, readout)

    #     test_utils = ModuleTester(model)

    #     for batch in dataloader:
    #         test_utils.test_shape((batch,), (n_batch, 1, 16))
