import molpot as mpot
import torch

from molpot.potentials.base import NNPotential
from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise
from molpot.trainer.metric.metrics import Identity, MAE
from molpot.trainer.logger.adapter import ConsoleHandler
from molpot import alias


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.QM9(data_dir="data/qm9", total=1000, batch_size=32)
    dp = qm9_dataset.prepare()
    train, valid = (
        dp.atomic_dress([1, 6, 7, 8, 9], alias.Z, alias.QM9.U)
        .calc_nblist(5)
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )
    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9

    n_atom_basis = 128
    arch = mpot.PaiNN(n_atom_basis, 3, GaussianRBF(20, 5), CosineCutoff(5))
    readout = Atomwise(n_in=n_atom_basis, output_key=alias.ti)
    model = NNPotential("PaiNN", arch, readout)
    criterion = mpot.MultiMSELoss([1], targets=[(alias.ti, alias.QM9.U)])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    stagnation = mpot.strategy.Stagnation(alias.loss, patience=torch.inf)

    mae = mpot.metric.MAE("energy_mae", alias.ti, alias.QM9.U)

    trainer = mpot.Trainer(
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
                "step": Identity(alias.step),
                "epoch": Identity(alias.epoch),
                "loss": Identity(alias.loss),
                "energy_mae": MAE(alias.ti, alias.QM9.U),
            },
            "handlers": [ConsoleHandler()],
        },
        config={
            "save_dir": "data/qm9",
            "device": {"type": "cpu"},
            "report_rate": 10,
            "valid_rate": 1000,
            "modify_lr_rate": 100,
        },
    )
    # trainer.jit()
    trainer.train(10000)
    return "done"


if __name__ == "__main__":
    train_qm9(load_qm9())
