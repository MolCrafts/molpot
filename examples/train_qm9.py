import os
from pathlib import Path
import molpot as mpot
import torch

from molpot.potentials.base import NNPotential
from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise
from molpot.trainer.metric.metrics import Identity, MAE
from molpot.trainer.logger.adapter import ConsoleHandler, TensorBoardHandler
from molpot import alias


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.QM9(data_dir="qm9", batch_size=64, total=1000)
    dp = qm9_dataset.prepare()
    train, valid = dp.calc_nblist(5).random_split(
        weights={"train": 0.8, "valid": 0.2}, seed=42
    )
    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9
    n_atom_basis = 16
    arch = mpot.potentials.nnp.PaiNN(
        n_atom_basis, 3, GaussianRBF(20, 5), CosineCutoff(5)
    )
    readout = Atomwise(16, input_key=alias.T0, output_key=alias.energy)
    model = NNPotential("PaiNN", arch, readout)
    criterion = mpot.MultiMSELoss([1], targets=[(alias.energy, alias.QM9.U)])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    stagnation = mpot.strategy.Stagnation(alias.loss, patience=torch.inf)

    mae = mpot.metric.MAE("energy_mae", alias.energy, alias.QM9.U)

    trainer = mpot.Trainer(
        "painn-qm9",
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        strategies=[stagnation],
        metrics=[mae],
        logger={
            "metrics": {
                "speed": Identity("speed"),
                "loss": Identity(alias.loss),
                "energy_mae": MAE(alias.energy, alias.QM9.U),
            },
            "handlers": [ConsoleHandler(), TensorBoardHandler()],
            "save_dir": "./log",
        },
        config={
            "resume": "model/painn-qm9.pt",
            "save_dir": "model",
            "device": {"type": "gpu", "n_gpu_use": 1},
            "compile": True,
            "report_rate": 10,
            "valid_rate": 10,
            "modify_lr_rate": 100,
            "checkpoint_rate": 5,
        },
    )

    trainer.train(10)

    return "done"


if __name__ == "__main__":
    proj_dir = Path.cwd() / "train_qm9"
    proj_dir.mkdir(exist_ok=True)
    os.chdir(proj_dir)
    train_qm9(load_qm9())
