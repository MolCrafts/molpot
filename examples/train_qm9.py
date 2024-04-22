import os
from pathlib import Path

import torch

import molpot as mpot
from molpot import Alias
from molpot.potential.nnp.readout import Atomwise


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.dataset.QM9(
        save_dir="qm9", batch_size=64, total=1000, device="cpu"
    )
    dp = qm9_dataset.prepare()
    # dp = dp.atomic_dress(types_list=[1, 6, 7, 8, 9], key=Alias.Z, prop=Alias.qm9.U0, buffer=1000)
    train, valid = dp.calc_nblist(5).random_split(
        weights={"train": 0.8, "valid": 0.2}, seed=43, total_length=1000
    )
    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9
    n_atom_basis = 10
    arch = mpot.potential.nnp.PaiNN(
        n_atom_basis, 4, mpot.nnp.GaussianRBF(n_atom_basis, 5), mpot.nnp.CosineCutoff(5)
    )
    readout = Atomwise(n_atom_basis, 1, input_key=Alias.painn.scalar, output_key=Alias.energy, aggregation_mode='add')
    model = mpot.PotentialSeq(arch, readout)
    criterion = mpot.trainer.loss.multi_targets(torch.nn.MSELoss(), [1], [(Alias.energy, Alias.qm9.U0)])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

    trainer = mpot.Trainer(
        "train_qm9",
        model,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        enable_amp=False
    )
    trainer.register_fix(
        mpot.trainer.io.CheckPointFix(50, 0, 5)
    )
    trainer.register_fix(
        mpot.trainer.io.TensorBoardFix(20, 0, tb_log_dir="tb_log")
    )
    trainer.register_fix(
        mpot.trainer.metric.MAE(20, mpot.Alias.energy, mpot.Alias.qm9.U0)
    )

    output = trainer.train(100)
    return "done"


if __name__ == "__main__":
    train_qm9(load_qm9())
