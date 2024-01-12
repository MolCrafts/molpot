import molpot as mpot
from molpot.potentials.base import NNPotential
import torch

from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise

from molpot import alias
from tests.conftest import batch_size

def load_rmd17() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    rmd17_dataset = mpot.rMD17(data_dir="data/rmd17", total=100, batch_size=32)
    dp = rmd17_dataset.prepare()
    train, valid = (
        dp.calc_nblist(5)
        .shuffle()
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )
    train_dataloader = mpot.create_dataloader(
        train.batch(batch_size=4).collate_data(), nworkers=0
    )
    valid_dataloader = mpot.create_dataloader(
        valid.batch(batch_size=16).collate_data(), nworkers=0
    )
    return train_dataloader, valid_dataloader


def train_rmd17(load_rmd17: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_rmd17

    n_atom_basis = 128
    model = NNPotential("PaiNN")
    arch = mpot.PaiNN(n_atom_basis, 3, GaussianRBF(20, 5), CosineCutoff(5))
    readout = Atomwise(n_in=n_atom_basis, output_key='_pred_energy')
    model.append(arch)
    model.append(readout)
    # TODO: _pred_energy -> alias
    # e.g. alias.scalar
    criterion = mpot.MultiMSELoss([1], targets=[("_pred_energy", alias.rMD17.U)])
    optimizer = torch.optim.Adam(model.parameters())

    trainer = mpot.Trainer(
        model,
        criterion,
        optimizer,
        train_dataloader,
        valid_data_loader=valid_dataloader,
        config={
            "save_dir": "data/rmd17",
            "metrics": [],
            "device": {"type": "cpu"},
        },
    )
    trainer.train(1e6)
    return "done"

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    train_rmd17(load_rmd17())