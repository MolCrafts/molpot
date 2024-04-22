import shutil

import pytest
import torch.nn as nn

import molpot as mpot


@pytest.fixture(scope='session')
def mocked_trainer():

    class EmptyModel(nn.Module):
        def forward(self, x):
            return x

    def _train_impl(self, data):
        self.train_result = {"steps": 1, "epochs": 1}
    Trainer = mpot.Trainer
    Trainer.train_impl = _train_impl

    trainer = Trainer(
        "test_trainer",
        model=EmptyModel(),
        loss_fn=lambda x, y: None,
        optimizer=nn.MSELoss(),
        lr_scheduler=None,
        dataloader=range(10),
        enable_amp=False
    )
    yield trainer
    # del work_dir
    work_dir = trainer.work_dir
    shutil.rmtree(work_dir)