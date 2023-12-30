import molpot as mpot
from molpot.potentials.base import NNPotential
import torch

from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise

from molpot import alias


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.QM9(data_dir="data/qm9", total=1000)
    dp = qm9_dataset.prepare()
    train, valid = (
        dp.calc_nblist(5)
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )
    train_dataloader = mpot.create_dataloader(train.batch(32))
    valid_dataloader = mpot.create_dataloader(valid.batch(1))
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9

    n_atom_basis = 128
    arch = mpot.PaiNN(n_atom_basis, 3, GaussianRBF(20, 5), CosineCutoff(5))
    readout = Atomwise(n_in=n_atom_basis, output_key=alias.ti)
    model = NNPotential("PaiNN", arch, readout)
    criterion = mpot.MultiMSELoss([1], targets=[(alias.ti, alias.QM9.U)])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    stagnation = mpot.strategy.Stagnation('loss', patience=torch.inf)

    mae = mpot.metric.MAE("energy_mae", alias.ti, alias.QM9.U)

    train_logger = mpot.logger.LogAdapter(
        "train_qm9",
        keys=["loss", "energy_mae"],
        # save_dir="data/qm9",
    )

    valid_logger = mpot.logger.LogAdapter(
        "valid_qm9",
        keys=["loss", "energy_mae"],
        # save_dir="data/qm9",
        is_echo=False,
    )

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
            "train": [train_logger],
            "valid": [valid_logger],
        },
        config={
            "save_dir": "data/qm9",
            "device": {"type": "cpu"},
            "n_train_log": 1000,
            # 'n_valiad_log': 10,
            "n_valid": 10,
            "n_lr_step": 100,
        },
    )
    trainer.train(1e6)
    return "done"


if __name__ == "__main__":
    train_qm9(load_qm9())
