import pytest
import torch
from molpot import Config, alias
import molpot as mpot
import tensordict as td

torch.use_deterministic_algorithms(True)

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
        dtype=p3.dtype,
    )
    return torch.einsum("ix...,xy->iy...", p3, rot)


class ModuleTester:
    def __init__(self, model):
        self.model = model

    def test_shape(self, input_data, expected_output_shape):
        output = self.model(*input_data)
        assert (
            output.shape == expected_output_shape
        ), f"Expected shape {expected_output_shape}, but got {output.shape}"

    def __call_model(self, model, inputs):
        if isinstance(inputs, td.TensorDict):
            return model(inputs)
        elif isinstance(inputs, (list, tuple)):
            return model(*inputs)
        else:
            return model(inputs)

    def test_equivariance(
        self, transform_input, transform_output, input_data, output_key=None
    ):
        original_output = transform_output(self.__call_model(self.model, input_data))

        transformed_input = transform_input(input_data)
        transformed_output = self.__call_model(self.model, transformed_input)

        if output_key is not None:
            result = transformed_output[output_key]
        else:
            result = transformed_output

        abs_diff = torch.abs(original_output - result)
        max_abs_diff = torch.max(abs_diff)
        rel_diff = abs_diff / (torch.abs(original_output) + 1e-10)
        max_rel_diff = torch.max(rel_diff)
        assert original_output.shape == result.shape, f"Shape mismatch: {original_output.shape} vs {result.shape}"
        assert torch.allclose(
            original_output/torch.linalg.norm(original_output, axis=1, keepdim=True), result/torch.linalg.norm(result, axis=1, keepdim=True), atol=1e-2, rtol=1e-2
        ), f"Model is not equivariant. Max absolute difference: {max_abs_diff:.2e}, Max relative difference: {max_rel_diff:.2e}"

    def test_invariance(self, transform_input, input_data, output_key=None):
        original_output = self.model(input_data)
        transformed_input = transform_input(input_data)
        transformed_output = self.model(transformed_input)
        if output_key is not None:
            desire = transformed_output[output_key]
            expect = original_output[output_key]
        else:
            desire = transformed_output
            expect = original_output
        assert torch.allclose(
            expect, desire, atol=1e-2, rtol=1e-2
        ), "Model is not invariant"


class TestPiNet:

    @pytest.fixture
    def config(self):
        return {
            "n_atoms": 5,
            "n_features": 4,
            "n_basis": 10,
            "n_frames": 9,
            "n_batch": 3,
        }

    @pytest.fixture
    def n_atoms(self, config):
        return config["n_atoms"]

    @pytest.fixture
    def n_features(self, config):
        return config["n_features"]

    @pytest.fixture
    def n_basis(self, config):
        return config["n_basis"]

    @pytest.fixture
    def n_frames(self, config):
        return config["n_frames"]

    @pytest.fixture
    def n_batch(self, config):
        return config["n_batch"]

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

        def rotate_input(data):
            (p3, pair_i, pair_j) = data
            return (rot3d(p3, axis, theta=theta), pair_i, pair_j)

        def rotate_output(data):
            return rot3d(data, axis, theta=theta)

        test_utils.test_equivariance(rotate_input, rotate_output, (p3, pair_i, pair_j))

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

    def test_eqvarlayer(self, gen_homogenous_frames, n_frames, n_batch, n_features):
        from molpot.potential.nnp.pinet import EqvarLayer
        from molpot.pipeline.dataloader import _compact_collate
        frames = _compact_collate(gen_homogenous_frames(n_frames))
        n_atoms = frames[alias.xyz].size(0)
        pair_i = frames[alias.pair_i]
        pair_j = frames[alias.pair_j]
        diff = frames[alias.pair_diff]
        n_pairs = pair_i.size(0)
        i1 = torch.rand((n_pairs, 1, n_features))
        p3 = torch.zeros(n_atoms, 3, n_features)
        eqvarl = EqvarLayer([n_features, n_features])
        output = eqvarl(p3, pair_i, pair_j, diff, i1)

        assert (output[0].shape == (n_atoms, 3, n_features))
        assert (output[1].shape == (n_pairs, 3, n_features))

        axis = torch.rand(3)
        theta = torch.rand(1)

        transformed_output = eqvarl(p3, pair_i, pair_j, rot3d(diff, axis, theta), i1)
        assert torch.allclose(rot3d(output[0], axis, theta), transformed_output[0], atol=1e-2, rtol=1e-2)


    def test_pinet3(self, gen_homogenous_frames, n_frames, n_batch, n_basis, n_features):
        r_cutoff = 5.0
        frames = gen_homogenous_frames(n_frames)
        dataloader = mpot.DataLoader(frames, batch_size=n_batch, shuffle=False)
        pinet = mpot.potential.nnp.PiNet(
            depth=5,
            basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
            cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
            pp_nodes=[n_features, n_features],
            pi_nodes=[n_features, n_features],
            ii_nodes=[n_features, n_features],
        )
        readout = mpot.potential.nnp.readout.Atomwise(
            [n_features, 1],
            in_keys=[("pinet", "p1"), alias.atom_batch],
            out_keys=("predict", "energy"),
        )

        model = mpot.PotentialSeq(pinet, readout)
        model.to(dtype=torch.float64)

        axis = torch.rand(3)
        theta = torch.rand(1)

        def rotate_input(batch):
            batch[alias.xyz] = rot3d(batch[alias.xyz], axis, theta)
            # recalculate pair_diff
            batch[alias.pair_diff] = batch[alias.xyz][batch[alias.pair_j]] - batch[alias.xyz][batch[alias.pair_i]]
            return batch

        for batch in dataloader:
            
            batch['pairs', 'diff'].to(dtype=torch.float64)
            original_output = model(batch)
            original_p1 = original_output['pinet', 'p1']
            original_p3 = original_output['pinet', 'p3']
            rotated_p3 = rot3d(original_p3, axis, theta)
            rotated_output = model(rotate_input(batch))
            expect_p1 = rotated_output['pinet', 'p1']
            expect_p3 = rotated_output['pinet', 'p3']
            assert torch.allclose(original_p1, expect_p1, atol=1e-2, rtol=1e-2, equal_nan=True), "p1 is not invariant"
            assert torch.allclose(rotated_p3, expect_p3, atol=1e-2, rtol=1e-2, equal_nan=True), "p3 is not invariant"

    def test_pinet1(self, gen_homogenous_frames, n_frames, n_batch, n_basis):
        r_cutoff = 5.0
        frames = gen_homogenous_frames(n_frames)
        dataloader = mpot.DataLoader(frames, batch_size=n_batch, shuffle=False)
        pinet = mpot.potential.nnp.PiNet(
            depth=4,
            basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
            cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
        )
        readout = mpot.potential.nnp.readout.Atomwise(
            [16, 1],
            in_keys=[("pinet", "p1"), alias.atom_batch],
            out_keys=("predict", "energy"),
        )

        model = mpot.PotentialSeq(pinet, readout)

        test_utils = ModuleTester(model)

        def permute_input(batch):
            perm = torch.randperm(batch[alias.xyz].size(0))
            batch[alias.xyz] = batch[alias.xyz][perm]
            batch[alias.atom_batch] = batch[alias.atom_batch][perm]
            return batch

        for batch in dataloader:
            test_utils.test_invariance(permute_input, input_data=batch, output_key=("pinet", "p1"))
