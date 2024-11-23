import pytest
import torch
from molpot import Config, alias
import molpot as mpot


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


class TestPiNet:

    @pytest.fixture
    def n_atoms(self):
        return 5

    @pytest.fixture
    def n_pairs(self, n_atoms):
        return torch.randint(n_atoms, n_atoms**2, ())

    @pytest.fixture
    def n_features(self):
        return 32

    @pytest.fixture
    def n_basis(self):
        return 10

    @pytest.fixture
    def p1(self, n_atoms, n_features):
        return torch.rand(n_atoms, 1, n_features, dtype=Config.ftype)
    
    @pytest.fixture
    def p3(self, n_atoms, n_features):
        return torch.rand(n_atoms, 3, n_features, dtype=Config.ftype)

    @pytest.fixture
    def rotate(self, p1):
        pass

    @pytest.fixture
    def pair_i(self, n_atoms, n_pairs):
        return torch.randint(0, n_atoms, (n_pairs,), dtype=torch.int64)

    @pytest.fixture
    def pair_j(self, n_atoms, n_pairs):
        return torch.randint(0, n_atoms, (n_pairs,), dtype=torch.int64)

    @pytest.fixture
    def basis(self, n_pairs, n_basis):
        return torch.rand(n_pairs, n_basis, dtype=Config.ftype)

    @pytest.fixture
    def i1(self, p1, pair_i):
        return p1[pair_i]

    @pytest.fixture
    def p3(self, n_atoms, n_features):
        return torch.rand(n_atoms, 3, n_features, dtype=Config.ftype)
    
    @pytest.fixture
    def n_frames(self):
        return 9
    
    @pytest.fixture
    def n_batch(self):
        return 3

    def test_fflayer(self, n_atoms, n_features, p1):
        from molpot.potential.nnp.pinet import FFLayer

        ppl = FFLayer(n_features, n_features, n_features * 2)
        out = ppl(p1)
        assert out.shape == (n_atoms, 1, n_features * 2)

    def test_pilayer(self, p1, pair_i, pair_j, basis, n_pairs, n_features, n_basis):
        from molpot.potential.nnp.pinet import PILayer

        pil = PILayer(n_features, n_features * n_basis)
        out = pil(p1, pair_i, pair_j, basis)
        assert out.shape == (n_pairs, 1, n_features)

    def test_iplayer(self, p1, n_atoms, i1, pair_i, n_features):
        from molpot.potential.nnp.pinet import IPLayer

        ipl = IPLayer()

        out = ipl(i1, pair_i, p1)
        assert out.shape == (n_atoms, 1, n_features)

    def test_pixlayer(self, p3, pair_i, pair_j, n_features):
        from molpot.potential.nnp.pinet import PIXLayer

        pixl = PIXLayer(n_features, n_features)
        out = pixl(p3, pair_i, pair_j)

        axis = torch.rand(3)
        theta = torch.rand(1)

        rot_p3 = rot3d(p3, axis, theta=theta)
        rot_out = pixl(rot_p3, pair_i, pair_j)
        assert torch.allclose(
            rot3d(out, axis, theta=theta),
            rot_out,
            atol=1e-6,
            rtol=1e-6,
        )  # float32

    def test_scalelayer(self, p3, p1, ):
        from molpot.potential.nnp.pinet import ScaleLayer
        scalel = ScaleLayer()
        out = scalel(p3, p1)
        assert out.shape == p3.shape

    def test_selfdotlayer(self, p3):
        from molpot.potential.nnp.pinet import SelfDotLayer
        sdl = SelfDotLayer()
        out = sdl(p3)
        assert out.shape == (p3.shape[0], 1, p3.shape[-1])

    @pytest.mark.parametrize("rank", [1, 3])
    def test_pinet(self, rank, gen_homogenous_frames, n_frames, n_batch, n_basis):
        r_cutoff = 5.0
        frames = gen_homogenous_frames(n_frames)
        dataset = mpot.Dataset("test_ds", frames)
        dataloader = mpot.DataLoader(dataset, batch_size=n_batch)
        pinet = mpot.potential.nnp.PiNet(
            depth=4,
            basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
            cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
            rank=rank,
        )
        readout = mpot.potential.nnp.readout.Atomwise(16, 1, from_key=("pinet", "p1"), to_key=("predict", "energy"))

        axis = torch.rand(3)
        theta = torch.rand(1)

        for batch in dataloader:
            batch1 = batch.copy()
            pinet(batch1)
            readout(batch1)
            energy1 = batch1["predict"]["energy"]
            
            batch2 = batch.copy()
            batch2[alias.xyz] = rot3d(batch2[alias.xyz], axis, theta)
            pinet(batch2)
            readout(batch2)
            energy2 = batch2["predict"]["energy"]
            assert torch.allclose(energy1, energy2, atol=1e-6, rtol=1e-6)

