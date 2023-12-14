import molpot as mpot
import torch

from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF


def load_qm9() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    qm9_dataset = mpot.QM9(data_dir="data/qm9", test_size=100)
    dp = qm9_dataset.prepare()
    train, valid = (
        dp.calc_nblist(5)
        .shuffle()
        .set_length(1000)
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )
    train_dataloader = mpot.create_dataloader(
        train.batch(batch_size=4).collate_data(), nworkers=0
    )
    valid_dataloader = mpot.create_dataloader(
        valid.batch(batch_size=16).collate_data(), nworkers=0
    )
    return train_dataloader, valid_dataloader


def train_qm9(load_qm9: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_qm9

    model = mpot.PaiNN(128, 3, GaussianRBF(20, 5), CosineCutoff(5))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    trainer = mpot.Trainer(
        model,
        criterion,
        optimizer,
        train_dataloader,
        valid_data_loader=valid_dataloader,
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