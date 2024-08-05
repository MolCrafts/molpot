import torch

import pytest
import molpot as mpot
from molpot.pipline.locality import TorchNeighborList, PairwiseDistances

from torch_scatter import scatter_add

import numpy.testing as npt


class TestLJ126:

    @pytest.fixture(scope="class", name="lj126")
    def test_init(self):

        from molpot.potential.classic.pair.lj import LJ126

        lj126 = LJ126(eps=1.0, sig=1.0)

        return lj126

    @pytest.fixture(scope="class", name="frame")
    def test_frame(self):

        n_atoms = 10

        box = mpot.Box(torch.eye(3) * 10)

        r_ij = torch.rand((n_atoms, 3)) * box.lengths
        # r_ij = torch.tensor([[0., 0., 0.], [1.0, 0., 0.]])
        r_ij.requires_grad_(True)

        inputs = mpot.Frame()
        inputs["atoms"]["xyz"] = r_ij
        inputs["box"]["cell"] = box.matrix
        inputs["box"]["pbc"] = torch.tensor([True, True, True])

        outputs = mpot.Frame()

        nblist = TorchNeighborList(cutoff=3.0)
        dist = PairwiseDistances()

        nblist(inputs, outputs)
        dist(inputs, outputs)

        inputs["n_atoms"] = n_atoms
        outputs["atoms"]['xyz'] = r_ij
        outputs["n_pairs"] = outputs["pairs"]["pair_dist"].shape[0]
        outputs["atoms"]["energy"] = torch.zeros(n_atoms)
        outputs["atoms"]["force"] = torch.zeros(n_atoms, 3)

        return inputs, outputs

    def test_energy(self, lj126, frame):
        inputs, outputs = frame

        inputs, outputs = lj126.energy(inputs, outputs)

        assert outputs["pairs"]["lj126_energy"].shape == (outputs["n_pairs"],)

    def test_forces(self, lj126, frame):
        inputs, outputs = frame
        inputs, outputs = lj126.forces(inputs, outputs)

        assert outputs["pairs"]["lj126_forces"].shape == (outputs["n_pairs"], 3)

    def test_forward(self, lj126, frame):
        inputs, outputs = frame
        inputs, outputs = lj126.forward(inputs, outputs)
        inputs, outputs = lj126.forces(inputs, outputs)

        assert outputs["pairs"]["lj126_energy"].shape == (outputs["n_pairs"],)
        # assert outputs['pairs']['lj126_forces'].shape == (frame['n_pairs'], 3)

        from torch.autograd import grad

        Epred = outputs["pairs"]["lj126_energy"]

        go = [torch.ones_like(Epred)]
        grads = grad(
            [Epred], outputs["atoms"]["xyz"], grad_outputs=go, create_graph=False
        )
        dEdR = grads[0]
        assert dEdR.shape == (inputs["n_atoms"], 3)
        npt.assert_allclose(
            dEdR.detach().numpy() / 2,
            scatter_add(
                outputs["pairs"]["lj126_forces"], outputs["pairs"]["pair_i"], dim=0
            )
            .detach()
            .numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_call_twice(self, lj126, frame):

        inputs, outputs = frame
        inputs, outputs = lj126.forward(inputs, outputs)
        inputs, outputs = lj126.forward(inputs, outputs)
