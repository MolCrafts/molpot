import molpot as mpot
import torch
from molpot import alias


class TestDerivative:

    # def test_bond_force(self, gen_homogenous_frames):

    #     r0 = torch.tensor(
    #         [
    #             [0.0, 0.9572],
    #             [0.9572, 0.0],
    #         ]
    #     )
    #     k = torch.tensor(
    #         [
    #             [0.0, 450],
    #             [450, 0.0],
    #         ]
    #     )

    #     bond_harmonic = mpot.potential.classic.bond.Harmonic(r0=r0, k=k)
    #     f_readout = mpot.potential.nnp.readout.DBondPot(
    #         fx_key=("predicts", "bond_harmonic_energy"),
    #         dx_key=("bonds", "dist"),
    #         to_key=("predicts", "bond_harmonic_forces"),
    #     )
    #     potential = mpot.PotentialSeq("test_derivative", bond_harmonic, f_readout)

    #     frame = gen_homogenous_frames(1)[0]
    #     result = potential(frame)

    #     type_id = frame["atoms", "type"]
    #     bondtype_i = type_id[frame["bonds", "i"]]
    #     bondtype_j = type_id[frame["bonds", "j"]]

    #     bond_forces = (
    #         (k[bondtype_i, bondtype_j]
    #         * (frame["bonds", "dist"] - r0[bondtype_i, bondtype_j]))[:, None]
    #         * (frame["bonds", "diff"] / frame["bonds", "dist"][:, None])
    #     )

    #     expected_forces = torch.zeros(
    #         (frame["atoms", "R"].shape[0], 3), dtype=bond_forces.dtype
    #     )

    #     expected_forces.index_add_(0, frame["bonds", "i"], -bond_forces)

    #     assert torch.allclose(
    #         result["predicts", "bond_harmonic_forces"], expected_forces
    #     )

    def test_pair_force(self, gen_homogenous_frames):

        frame = gen_homogenous_frames(1)[0]
        frame = mpot.process.nblist.NeighborList(2.0)(frame)


        # calculate expected energy and forces with coordinates
        R = frame["atoms", "R"].requires_grad_(True)
        pair_energy = (
            0.5
            * torch.linalg.norm(R[frame[alias.pair_j]] - R[frame[alias.pair_i]], dim=1)
            ** 2
        )
        pair_force = -1 * torch.autograd.grad(
            pair_energy.sum(), R, create_graph=True, retain_graph=True
        )[0]

        # molpot uses pair vector as input of all potentials
        frame["predicts", "pair_energy"] = torch.sum(
            0.5 * torch.linalg.norm(frame["pairs", "diff"], dim=1) ** 2
        )
        force_readout = mpot.potential.nnp.base.PairForce(
            in_key=("predicts", "pair_energy"),
            dx_key=("pairs", "diff"),
            out_key=("predicts", "forces"),
        )
        forces = force_readout(
            torch.sum(frame["predicts", "pair_energy"]),
            frame["pairs", "diff"],
            frame["pairs", "i"],
            frame["pairs", "j"],
        )

        # compare the results,
        # mainly test if the direction of forces is correct
        assert torch.allclose(forces, pair_force)
