import molpot as mpot
import pytest
import numpy as np
import numpy.testing as npt
def test_get_potential():

    kernel = mpot.potential.get_classic_potental('LJ126', 'pair')
    assert kernel.__name__ == 'LJ126'

class TestForceField:

    @pytest.fixture(scope='class', name='ff')
    def test_init(self):
        ff = mpot.ForceField()

        return ff
    
    def test_def_atom(self, ff:mpot.ForceField):

        atomstyle = ff.def_atomstyle("atomic")
        atomstyle.def_atomtype("O", 0, mass=15.9994)
        atomstyle.def_atomtype("H", 1, mass=1.00794)
    
    def test_def_pair(self, ff:mpot.ForceField):

        pairstyle = ff.def_pairstyle("LJ126", global_cutoff=10.0, mixing="arithmetic")
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

    def test_bond(self, ff: mpot.ForceField):

        bondstyle = ff.def_bondstyle(
            "Harmonic",
        )
        bondstyle.def_bondtype(0, 1, r0=1.012, k=1059.162, name="O-H")
        params = bondstyle.get_params()
        npt.assert_allclose(params['r0'], np.array([[0, 1.012], [1.012, 0]]))
        assert bondstyle.n_types == 1  # O-H, H-O

    def test_get_potential(self, ff:mpot.ForceField):

        pot = ff.get_potential()
        assert len(pot) == 2

