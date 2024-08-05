import pytest
import numpy as np
import numpy.testing as npt
import molpot as mpot
from molpot.utils.forcefield import Style, Type


class TestStyle:

    def test_init_with_potential(self):

        style = Style(mpot.classic.bond.Harmonic)
        assert style.name == "BondHarmonic"


class TestForceField:

    @pytest.fixture(scope="class", name="ff")
    def init_forcefield(self):

        ff = mpot.ForceField()
        return ff

    def test_atom(self, ff: mpot.ForceField):

        atomstyle = ff.def_atomstyle("atomic")
        atomstyle.def_atomtype("O", 0, mass=15.9994)
        atomstyle.def_atomtype("H", 1, mass=1.00794)

        assert atomstyle.n_types == 2

    def test_bond(self, ff: mpot.ForceField):

        bondstyle = ff.def_bondstyle(
            mpot.classic.bond.Harmonic,
        )
        bondstyle.def_bondtype(0, 1, r0=1.012, k=1059.162, name="O-H")
        params = bondstyle.get_params()
        npt.assert_allclose(params["r0"], np.array([[0, 1.012], [1.012, 0]]))
        assert bondstyle.n_types == 1  # O-H, H-O

    def test_angle(self, ff: mpot.ForceField):

        anglestyle = ff.def_anglestyle(mpot.classic.angle.Harmonic)
        anglestyle.def_angletype(1, 0, 1, theta0=104.52, k=75.90, name="H-O-H", )

        n_atomtypes = ff.n_atomtypes
        n_angletype = anglestyle.n_types
        assert n_angletype == 1, ValueError(f"Expected 2 atom types, got {n_angletype}")
        theta0 = anglestyle.get_params()['theta0']
        assert theta0.shape == (n_atomtypes, n_atomtypes, n_atomtypes)

        expected_theta0 = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        expected_theta0[1, 0, 1] = 104.52

        npt.assert_equal(theta0, expected_theta0)

    def test_pair(self, ff: mpot.ForceField):

        pairstyle = ff.def_pairstyle(
            mpot.classic.pair.LJ126, global_cutoff=10.0, mixing="arithmetic"
        )
        pairstyle.def_pairtype("O-O", 0, 0, eps=0.1553, sig=3.1506)
        pairstyle.def_pairtype("O-H", 0, 1, eps=0.0, sig=1.0)
        pairstyle.def_pairtype("H-H", 1, 1, eps=0.0, sig=1.0)

        params = pairstyle.get_params()
        npt.assert_allclose(
            params["eps"],
            np.array([[0.1553, 0.0], [0.0, 0.0]]),
        )
        npt.assert_allclose(
            params["sig"],
            np.array([[3.1506, 1.0], [1.0, 1.0]]),
        )

    def test_get_potential(self, ff:mpot.ForceField):

        potential = ff.get_potential()
        assert len(potential) == 3