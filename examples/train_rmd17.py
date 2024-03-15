import os
from pathlib import Path

import torch

import molpot as mpot
from molpot import Alias
from molpot.potentials.base import Potentials
from molpot.potentials.nnp.layers import CosineCutoff, GaussianRBF
from molpot.potentials.nnp.readout import Atomwise
from molpot.trainer.logger.adapter import ConsoleHandler, TensorBoardHandler
from molpot.trainer.metric.metrics import Identity


def load_rmd17() -> tuple[mpot.DataLoader, mpot.DataLoader]:
    rmd17_dataset = mpot.dataset.RMD17(save_dir="rmd17", batch_size=64, device="cpu", atom_dress=True, total=1000)
    dp = rmd17_dataset.prepare()
    dp = dp.atomic_dress(types_list=[1, 6, 8], key=Alias.Z, prop=Alias.rmd17.energy, buffer=1000)
    train, valid = dp.calc_nblist(4.5).random_split(
        weights={"train": 0.8, "valid": 0.2}, seed=42
    )
    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader


def train_rmd17(load_rmd17: tuple[mpot.DataLoader, mpot.DataLoader]) -> str:
    train_dataloader, valid_dataloader = load_rmd17
    n_atom_basis = 10
    n_atom_types = 3
    pp_nodes = [64, 64, 64, 64]
    pi_nodes = [64, 64]
    ii_nodes = [64, 64, 64, 64]
    arch = mpot.potentials.nnp.PiNet(n_atom_types,
        n_atom_basis, 5, GaussianRBF(n_atom_basis, 5), CosineCutoff(5),
        pp_nodes, pi_nodes, ii_nodes
    )
    # define the readout layers
    energy_readout = Atomwise(ii_nodes[-1], [64, 32, 16], 1, input_key=Alias.pinet.p1, output_key=Alias.energy)
    ## TODO: forces_readout = Atomwise(n_atom_basis, [], 3, input_key=Alias.T1, output_key=Alias.forces)

    model = mpot.NNPotential("pinet", arch, energy_readout, derive_energy=False)

    criterion = mpot.MultiMSELoss(
        [1],
        targets=[
            (Alias.energy, Alias.rmd17.energy),
            # (Alias.forces, Alias.rmd17.forces),
        ],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.994)

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
                "speed": mpot.metric.StepSpeed(),
                "loss": Identity(Alias.loss),
                "energy_mae": mpot.metric.MAE(
                    Alias.energy, Alias.rmd17.energy
                ),
                # "forces_mae": mpot.metric.MAE(
                #     Alias.forces, Alias.rmd17.forces
                # ),
            },
            "handlers": [ConsoleHandler(), TensorBoardHandler()],
            "save_dir": "./log",
        },
        config={
            "save_dir": "model",
            "device": {"type": "cpu"},
            "compile": False,
            "report_rate": 1,
            "valid_rate": 10000,
            "modify_lr_rate": 10000,
            "checkpoint_rate": 10000,
        },
    )

    output = trainer.train(1000)
    for data in train_dataloader:
        output = model(data)
    energy = output[Alias.energy]
    from torchviz import make_dot
    make_dot(energy, params=dict(model.named_parameters()), show_attrs=True, show_saved=True, ).render("pinet3", format="png")
    print("done")
    return "done"


if __name__ == "__main__":
    proj_dir = Path.cwd() / "train_rmd17"
    proj_dir.mkdir(exist_ok=True)
    os.chdir(proj_dir)
    train_rmd17(load_rmd17())
