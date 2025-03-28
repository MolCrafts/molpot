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


def assert_close(desired, expected, atol, rtol, msg):
    assert (
        desired.shape == expected.shape
    ), f"Expected shape {expected.shape}, but got {desired.shape}"
    abs_diff = torch.abs(expected - desired)
    max_abs_diff = torch.max(abs_diff)
    rel_diff = abs_diff / (torch.abs(desired) + 1e-10)
    max_rel_diff = torch.max(rel_diff)
    assert torch.allclose(
        desired, expected, atol=atol, rtol=rtol
    ), f"{msg}. Max absolute difference: {max_abs_diff:.2e}, Max relative difference: {max_rel_diff:.2e}"


axis = torch.rand(3)
theta = torch.rand(1)


class ModuleTester:
    def __init__(self, model):
        self.model = model

    def test_shape(self, input_data, expected_output_shape, compare_fn=lambda x: x):
        output = self.model(*input_data)
        output = compare_fn(output)
        assert (
            output.shape == expected_output_shape
        ), f"Expected shape {expected_output_shape}, but got {output.shape}"

    def _call_model(self, model, inputs):
        if isinstance(inputs, td.TensorDict):
            return model(inputs)
        elif isinstance(inputs, (list, tuple)):
            return model(*inputs)
        else:
            return model(inputs)

    def test_equivariance(
        self,
        transform_input,
        transform_output,
        input_data,
        compare_fn=lambda x: x,
        atol=1e-6,
        rtol=1e-6,
    ):
        original_output = self._call_model(self.model, transform_input(*input_data))

        transformed_output = transform_output(self._call_model(self.model, input_data))

        desired = compare_fn(original_output)
        expected = compare_fn(transformed_output)

        assert_close(desired, expected, atol, rtol, "Model is not equivariant")

    def test_invariance(
        self, transform_input, input_data, compare_fn=lambda x: x, atol=1e-6, rtol=1e-6
    ):
        output = self._call_model(self.model, input_data)
        original_output = self._call_model(self.model, transform_input(*input_data))
        desired = compare_fn(original_output)
        expected = compare_fn(output)
        assert_close(desired, expected, atol, rtol, "Model is not invariant")


class TestPiNet:

    @pytest.fixture
    def config(self):
        return {
            "n_atoms": 5,
            "n_features": 4,
            "n_basis": 10,
            "n_frames": 9,  # total frames generated
            "n_batches": 3,  # number of batches
        }

    @pytest.fixture
    def n_batches(self, config):
        return config["n_batches"]

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

    def test_fflayer(self, n_atoms, n_features, p1):
        from molpot.potential.nnp.pinet import FeedForward

        ppl = FeedForward(n_features, n_features, n_features * 2)
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

        def rotate_input(p3, pair_i, pair_j):
            return rot3d(p3, axis=axis, theta=theta), pair_i, pair_j

        def rotate_output(output):
            return rot3d(output, axis=axis, theta=theta)

        test_utils.test_equivariance(rotate_input, rotate_output, (p3, pair_i, pair_j))

    def test_scalelayer(self, p3, p1):
        from molpot.potential.nnp.pinet import ScaleLayer

        scalel = ScaleLayer()
        test_utils = ModuleTester(scalel)
        test_utils.test_shape((p3, p1), p3.shape)

    def test_selfdotlayer(self, p3):
        from molpot.potential.nnp.pinet import DotLayer

        sdl = DotLayer()
        test_utils = ModuleTester(sdl)
        test_utils.test_shape((p3,), (p3.shape[0], 1, p3.shape[-1]))

    def test_eqvarlayer(self, gen_homogenous_frames, n_frames, n_features):
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
        eqvarl = ModuleTester(EqvarLayer(n_features, n_features))
        eqvarl.test_shape(
            (p3, pair_i, pair_j, diff, i1), (n_pairs, 3, n_features), lambda x: x[0]
        )

        def rotate_input(p3, pair_i, pair_j, diff, i1):
            frames[alias.xyz] = rot3d(frames[alias.xyz], axis, theta)
            diff = (
                frames[alias.xyz][frames[alias.pair_j]]
                - frames[alias.xyz][frames[alias.pair_i]]
            )
            return p3, pair_i, pair_j, diff, i1

        def rotate_output(output):
            return rot3d(output[0], axis, theta), *output[1:]

        eqvarl.test_equivariance(
            rotate_input, rotate_output, (p3, pair_i, pair_j, diff, i1), lambda x: x[0]
        )

    def test_gcblock3(self, gen_homogenous_frames, n_frames, n_features):

        from molpot.potential.nnp.pinet import GCBlock3
        from molpot.pipeline.dataloader import _compact_collate

        frames = _compact_collate(gen_homogenous_frames(n_frames))
        n_atoms = frames[alias.xyz].size(0)

        p1 = torch.rand(n_atoms, 1, n_features)
        p3 = torch.zeros(n_atoms, 3, n_features)
        pair_i = frames[alias.pair_i]
        pair_j = frames[alias.pair_j]
        diff = frames[alias.pair_diff]
        basis_fn = mpot.potential.nnp.radial.GaussianRBF(10, 5.0)
        cutoff_fn = mpot.potential.nnp.cutoff.CosineCutoff(5.0)
        basis = basis_fn(torch.linalg.norm(diff, dim=-1))
        fc = cutoff_fn(torch.linalg.norm(diff, dim=-1))
        basis = basis * fc[..., None]

        gcblock = ModuleTester(
            GCBlock3(
                [n_features, n_features],
                [n_features, n_features * 10],
                [n_features, n_features],
            )
        )

        def rotate_input(p1, p3, pair_i, pair_j, basis, diff):
            xyz = rot3d(frames[alias.xyz], axis, theta)
            diff = xyz[frames[alias.pair_j]] - xyz[frames[alias.pair_i]]
            basis = basis_fn(torch.linalg.norm(diff, dim=-1))
            fc = cutoff_fn(torch.linalg.norm(diff, dim=-1))
            basis = basis * fc[..., None]
            return (p1, p3, pair_i, pair_j, basis, diff)

        def rotate_output(output):
            return (output[0][0], rot3d(output[0][1], axis, theta)), output[1]

        gcblock.test_equivariance(
            rotate_input,
            rotate_output,
            (p1, p3, pair_i, pair_j, basis, diff),
            lambda x: x[0][1],
        )

    def test_pinet3(
        self, gen_homogenous_frames, n_frames, n_batches, n_basis, n_features
    ):
        r_cutoff = 5.0
        frames = gen_homogenous_frames(n_frames)
        dataloader = mpot.DataLoader(frames, batch_size=n_batches, shuffle=False)
        model = ModuleTester(
            mpot.potential.nnp.PiNet2(
                depth=5,
                basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
                cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
                pp_nodes=[n_features, n_features],
                pi_nodes=[n_features, n_features],
                ii_nodes=[n_features, n_features]
            )
        )

        for batch in dataloader:
            Z = batch[alias.Z]
            xyz = batch[alias.xyz]
            pair_i = batch[alias.pair_i]
            pair_j = batch[alias.pair_j]
            diff = xyz[pair_j] - xyz[pair_i]

            def rotate_input(Z, diff, pair_i, pair_j):
                _xyz = rot3d(xyz, axis, theta)
                diff = _xyz[pair_j] - _xyz[pair_i]
                return (Z, diff, pair_i, pair_j)

            def rotate_output(output):
                return (output[0], rot3d(output[1], axis, theta), *output[2:])

            model.test_equivariance(
                rotate_input,
                rotate_output,
                input_data=(Z, diff, pair_i, pair_j),
                compare_fn=lambda x: x[1],
            )

    # def test_pinet1(self, gen_homogenous_frames, n_frames, n_batches, n_basis):
    #     r_cutoff = 5.0
    #     frames = gen_homogenous_frames(n_frames)
    #     dataloader = mpot.DataLoader(frames, batch_size=n_batches, shuffle=False)
    #     model = mpot.potential.nnp.PiNet1(
    #         depth=4,
    #         basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
    #         cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
    #         rank=1,
    #     )

    #     test_utils = ModuleTester(model)

    #     for batch in dataloader:
    #         Z = batch[alias.Z]
    #         xyz = batch[alias.xyz]
    #         diff = batch[alias.pair_diff]
    #         pair_i = batch[alias.pair_i]
    #         pair_j = batch[alias.pair_j]

    #         def rotate_input(Z, diff, pair_i, pair_j):
    #             _xyz = rot3d(xyz, axis, theta)
    #             pair_diff = _xyz[pair_j] - _xyz[pair_i]
    #             return Z, pair_diff, pair_i, pair_j
            
    #         test_utils.test_invariance(
    #             rotate_input, (Z, diff, pair_i, pair_j), compare_fn=lambda x: x[0]
    #         )
