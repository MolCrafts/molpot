import pytest
import numpy as np
import molpot as mpot
from pathlib import Path

@pytest.fixture
def test_qm9_path():
    path = Path(__file__).parent / "testdata/test_qm9"
    return path


# @pytest.mark.skip(
#     "Run only local, not in CI. Otherwise takes too long and requires downloading the data"
# )
def test_qm9(test_qm9_path):
    
    qm9_dataset = mpot.QM9(data_dir=test_qm9_path)
    dp = qm9_dataset.prepare()
    dp = dp.calc_nblist(5).batch(batch_size=3).collate_frames()
    dataloader = mpot.create_dataloader(dp, nworkers=0)
    for d in enumerate(dataloader):
        print(d)
        break