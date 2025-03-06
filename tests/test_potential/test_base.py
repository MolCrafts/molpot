import pytest
import torch
from molpot.potential.base import Potential, PotentialSeq
from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class TestBasePotential:

    @pytest.fixture(name="td")
    def mock_tensordict(self):
        return TensorDict(
            {
                "positions": torch.randn(10, 3),
                "charges": torch.randn(10),
            }
        )

    @pytest.fixture(name="fixed_keys_potential")
    def mock_fixed_keys_potential(self):
        class FixedKeysPotential(Potential):

            in_keys = ("positions", "charges")
            out_keys = ("energy",)

            def __init__(self):
                super().__init__()

            def forward(self, positions, charges):
                return True

        potential = FixedKeysPotential()
        return potential

    @pytest.fixture(name="dynamic_keys_potential")
    def mock_dynamic_keys_potential(self):

        class DynamicKeysPotential(Potential):

            def __init__(self, in_keys, out_keys):
                super().__init__()
                self.in_keys = in_keys
                self.out_keys = out_keys

            def forward(self, positions, charges):
                return True

        potential = DynamicKeysPotential(
            in_keys=("positions", "charges"), out_keys=("energy",)
        )

        return potential

    @pytest.mark.parametrize(
        "potential", ["fixed_keys_potential", "dynamic_keys_potential"]
    )
    def test_keys(self, request, potential, td):
        potential = request.getfixturevalue(potential)
        assert potential(td["positions"], td["charges"])
        td = TensorDictModule(
            potential, in_keys=potential.in_keys, out_keys=potential.out_keys
        )(td)
        assert set(td.keys()) == {"energy", "positions", "charges"}


class TestPotentialSeq:

    ...
