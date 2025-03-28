import molpot as mpot
import numpy as np
import numpy.testing as npt
import torch

def assert_allclose(a, b, rtol=1e-6, atol=1e-6):
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    npt.assert_allclose(a, b, rtol=rtol, atol=atol)

class TestUnit:

    def test_real_unit(self):
        unit = mpot.Unit("real")
        assert_allclose(
            unit.kB,
            0.001987  # kcal/mol
        )
        assert_allclose(
            unit.convert_unit("kcal/mol", "kJ/mol"),
            4.184
        )

    def test_electron_unit(self):
        unit = mpot.Unit("electron")
        assert_allclose(
            unit.kB,
            3.1668114e-6  # hartree
        )