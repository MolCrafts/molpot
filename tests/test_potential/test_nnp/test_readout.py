import molpot as mpot
import torch


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
        # Create a simple pair energy function that depends on distance
        frame = gen_homogenous_frames(1)[0]
        
        # Create some test pair energies (could be from a neural network)
        pair_diff = frame["pairs", "diff"].requires_grad_(True)
        pair_energies = 0.5 * torch.linalg.norm(frame["pairs", "diff"], dim=1) ** 2
        pair_force = -frame["pairs", "diff"]
        
        # Create the PairForce module
        force_readout = mpot.potential.nnp.readout.PairForce(
            in_keys=("predicts", "pair_energy"),
            dx_key=("pairs", "diff"),
            out_keys=("predicts", "forces")
        )
        
        # Calculate forces
        forces = force_readout(
            torch.sum(pair_energies),
            pair_diff,
            frame["pairs", "i"],
            frame["pairs", "j"]
        )
        
        # For unit energies, forces should point along the pair vectors
        # and have magnitude 1 for each pair
        expected_forces = torch.zeros_like(frame["atoms", "R"])
        # Accumulate forces on atoms i (negative direction)
        expected_forces.index_add_(
            0, 
            frame["pairs", "i"], 
            -pair_force
        )
        
        # Accumulate forces on atoms j (positive direction)
        expected_forces.index_add_(
            0,
            frame["pairs", "j"],
            pair_force
        )
        print(forces)
        print(expected_forces)
        assert torch.allclose(forces, expected_forces, atol=1e-6)

        
