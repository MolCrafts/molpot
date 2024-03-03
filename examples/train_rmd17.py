import os
from pathlib import Path

import torch

import molpot as mpot
from molpot import Alias
from molpot.potentials.base import NNPotential
from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise
from molpot.trainer.logger.adapter import ConsoleHandler, TensorBoardHandler
from molpot.trainer.metric.metrics import Identity


def load_rmd17() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    rmd17_dataset = mpot.dataset.RMD17(save_dir="rmd17", batch_size=64, total=1000, device="cuda")
    dp = rmd17_dataset.prepare()
    train, valid = dp.calc_nblist(5).random_split(
        weights={"train": 0.8, "valid": 0.2}, seed=42
    )
    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader


def train_rmd17(load_rmd17: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_rmd17
    n_atom_basis = 16
    arch = mpot.potentials.nnp.PiNetP3(
        n_atom_basis, 5, GaussianRBF(20, 5), CosineCutoff(5)
    )
    energy_readout = Atomwise(16, [], 1, input_key=Alias.T0, output_key='per_atom_energy')
    # forces_readout = Atomwise(16, 3, input_key=Alias.T1, output_key='per_atom_forces')
    model = NNPotential("pinet", arch, energy_readout) #, forces_readout)
    criterion = mpot.MultiMSELoss(
        [1, 1],
        targets=[
            (Alias.energy, Alias.rmd17.energy),
            (Alias.forces, Alias.rmd17.forces),
        ],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    stagnation = mpot.strategy.Stagnation(Alias.loss, patience=torch.inf)

    trainer = mpot.Trainer(
        "pinet-rmd17",
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
                    Alias.energy, Alias.rmd17.energy
                ),
                "forces_mae": mpot.metric.MAE(
                    Alias.forces, Alias.rmd17.forces
                ),
            },
            "handlers": [ConsoleHandler(), TensorBoardHandler()],
            "save_dir": "./log",
        },
        config={
            "save_dir": "model",
            "device": {"type": "gpu", "n_gpu_use": 1},
            "compile": True,
            "report_rate": 2,
            "valid_rate": 5,
            "modify_lr_rate": 5,
            "checkpoint_rate": 5,
        },
    )

    trainer.train(10)

    return "done"


if __name__ == "__main__":
    proj_dir = Path.cwd() / "train_rmd17"
    proj_dir.mkdir(exist_ok=True)
    os.chdir(proj_dir)
    train_rmd17(load_rmd17())
