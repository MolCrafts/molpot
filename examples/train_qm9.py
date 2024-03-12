import os
from pathlib import Path

import torch

import molpot as mpot
from molpot import Alias
from molpot.potentials.base import NNPotential
from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise
from molpot.trainer.logger.adapter import ConsoleHandler, TensorBoardHandler
from molpot.trainer.metric.metrics import MAE, Identity


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.dataset.QM9(
        save_dir="qm9", batch_size=10, total=1000, device="cpu"
    )
    dp = qm9_dataset.prepare()
    train, valid = dp.calc_nblist(5).random_split(
        weights={"train": 0.8, "valid": 0.2}, seed=42
    )
    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9
    n_atom_basis = 128
    arch = mpot.potentials.nnp.PaiNN(
        n_atom_basis, 3, GaussianRBF(n_atom_basis, 5), CosineCutoff(5)
    )
    readout = Atomwise(n_atom_basis, [64, 32, 16], 1, input_key=Alias.painn.p1, output_key=Alias.energy)
    model = NNPotential("PaiNN", arch, readout, derive_energy=True)
    criterion = mpot.MultiMSELoss([1], targets=[(Alias.energy, Alias.qm9.U0)])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    stagnation = mpot.strategy.Stagnation(Alias.loss, patience=torch.inf)

    trainer = mpot.Trainer(
        "painn-qm9",
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        strategies=[stagnation],
        logger={
            "metrics": {
                "speed": Identity("speed"),
                "loss": Identity(Alias.loss),
                "energy_mae": mpot.metric.MAE(
                    Alias.energy, Alias.qm9.U
                ),
            },
            "handlers": [ConsoleHandler(), TensorBoardHandler()],
            "save_dir": "./log",
        },
        config={
            "save_dir": "model",
            "device": {"type": "gpu"},
            "compile": False,
            "report_rate": 5,
            "valid_rate": 5,
            "modify_lr_rate": 5,
            "checkpoint_rate": 5,
        },
    )


    trainer.train(10000)

    return "done"


if __name__ == "__main__":
    proj_dir = Path.cwd() / "train_qm9"
    proj_dir.mkdir(exist_ok=True)
    os.chdir(proj_dir)
    train_qm9(load_qm9())
