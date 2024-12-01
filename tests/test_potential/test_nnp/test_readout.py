import molpot as mpot
import torch


class TestDerivative:

    def test_bond_force(self, gen_homogenous_frames):

        r0 = torch.tensor(
            [
                [0.0, 0.9572],
                [0.9572, 0.0],
            ]
        )
        k = torch.tensor(
            [
                [0.0, 450],
                [450, 0.0],
            ]
        )

        bond_harmonic = mpot.potential.classic.bond.Harmonic(r0=r0, k=k)
        f_readout = mpot.potential.nnp.readout.DBondPot(
            fx_key=("predicts", "bond_harmonic_energy"),
            dx_key=("bonds", "dist"),
            to_key=("predicts", "bond_harmonic_forces"),
        )
        potential = mpot.PotentialSeq("test_derivative", bond_harmonic, f_readout)

        frame = gen_homogenous_frames(1)[0]
        result = potential(frame)

        type_id = frame["atoms", "type"]
        bondtype_i = type_id[frame["bonds", "i"]]
        bondtype_j = type_id[frame["bonds", "j"]]

        bond_forces = (
            (k[bondtype_i, bondtype_j]
            * (frame["bonds", "dist"] - r0[bondtype_i, bondtype_j]))[:, None]
            * (frame["bonds", "diff"] / frame["bonds", "dist"][:, None])
        )

        expected_forces = torch.zeros(
            (frame["atoms", "R"].shape[0], 3), dtype=bond_forces.dtype
        )

        expected_forces.index_add_(0, frame["bonds", "i"], -bond_forces)

        assert torch.allclose(
            result["predicts", "bond_harmonic_forces"], expected_forces
        )

    def test_pair_force(self, gen_homogenous_frames):

        epsilon = torch.tensor(
            [
                [0.1521, 0.0836],
                [0.0836, 0.0460],
            ]
        )
        sigma = torch.tensor(
            [
                [3.1507, 1.7753],
                [1.7753, 0.4],
            ]
        )