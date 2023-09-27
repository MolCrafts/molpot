import pytest
import numpy as np
from molpot import QM9DataSet
from pathlib import Path

@pytest.fixture
def test_qm9_path():
    path = Path(__file__).parent / "testdata/test_qm9"
    return path


# @pytest.mark.skip(
#     "Run only local, not in CI. Otherwise takes too long and requires downloading the data"
# )
def test_qm9(test_qm9_path):
    qm9 = QM9DataSet(
        test_qm9_path,
        batch_size=5,
        n_train=10,
        n_valid=5,
        remove_uncharacterized=True,
    )
    qm9.prepare()
    loader = qm9.get_loader()