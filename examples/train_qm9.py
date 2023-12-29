import molpot as mpot
from molpot.potentials.base import NNPotential
import torch

from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise

from molpot import alias


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.QM9(data_dir="data/qm9", total=100)
    dp = qm9_dataset.prepare()
    train, valid = (
        dp.calc_nblist(5)
        .shuffle()
        .set_length(1000)
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )
    train_dataloader = mpot.create_dataloader(
        train.batch(batch_size=4)
    )
    valid_dataloader = mpot.create_dataloader(
        valid.batch(batch_size=16)
    )
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9

    n_atom_basis = 128
    model = NNPotential("PaiNN")
    arch = mpot.PaiNN(n_atom_basis, 3, GaussianRBF(20, 5), CosineCutoff(5))
    readout = Atomwise(n_in=n_atom_basis, output_key=alias.ti)
    model.append(arch)
    model.append(readout)
    criterion = mpot.MultiMSELoss([1], targets=[(alias.ti, alias.QM9.U)])
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    stagnation = mpot.strategy.Stagnation(alias.energy)

    train_acc = mpot.metric.Accuracy(alias.ti, alias.QM9.U)

    logger = mpot.trainer.logger.LogAdapter(
        "train_qm9", "data/qm9",
        keys=['loss', 'acc'],
    )

    trainer = mpot.Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        strategies=[stagnation],
        metrics=[train_acc],
        logger=logger,
        config={
            "save_dir": "data/qm9",
            "metrics": [],
            "device": {"type": "cpu"},
        },
    )
    trainer.train(1e6)
    return "done"


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    train_qm9(load_qm9())
