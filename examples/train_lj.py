import os
from pathlib import Path

import molexp as me
import molpy as mp
import torch

import molpot as mpot
from molpot import Alias
from molpot.pipline.dataloaders import DataLoader
from molpot.trainer.logger.adapter import ConsoleHandler
from molpot.trainer.metric.metrics import MAE, Identity


def gen_lj()->None:

    script = me.Script('lj.in')
    script.content = f"""
        units lj
        dimension 3
        atom_style atomic
        pair_style lj/cut 2.5
        boundary p p p
        region simulation_box block 0 10 0 10 0 10
        create_box 1 simulation_box
        create_atoms 1 random 100 341341 simulation_box
        # create_atoms 2 random 100 127569 simulation_box
        variable step equal step
        variable etotal equal etotal
        mass 1 1
        # mass 2 1
        pair_coeff 1 1 1.0 1.0
        # pair_coeff 2 2 0.5 3.0
        thermo 1000
        thermo_style custom step temp pe ke etotal press
        minimize 1.0e-4 1.0e-6 1000 10000
        fix mynve all nve
        fix mylgv all langevin 1.0 1.0 0.1 1530917
        dump 1 all custom 1000 lj.lammpstrj id type x y z
        fix prt all print 1000 "${{step}} ${{etotal}}" file lj.log screen no
        timestep 0.005
        run 5000
    """

    engine = me.engine.LAMMPSEngine('lmp')
    engine.add_script(script)
    engine.run('mpirun -np 4 lmp -in lj.in', cwd='tmp')
    return None


def load_lj(gen_lj)->tuple:

    traj = mp.io.load_trajectory('tmp/lj.lammpstrj')
    log = mp.io.loadtxt("tmp/lj.log")

    lj_dataset = mpot.dataset.Trajectory(traj, total=10)
    dp = lj_dataset.prepare()
    dp = dp.zip(log[:,0])
    train, valid = (
        dp.calc_nblist(2.5)
        .collate_data()
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )

    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader

def train_lj(load_lj: tuple[DataLoader, DataLoader]) -> str:

    train_dataloader, valid_dataloader = load_lj
    lj_pot = mpot.classical.pair.LJ126(1, 2.5)
    pot = mpot.Potentials(lj_pot)
    criterion = mpot.MultiMSELoss([1], targets=[(Alias.ti, Alias.energy)])
    optimizer = torch.optim.Adam(pot.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    stagnation = mpot.strategy.Stagnation(Alias.loss, patience=torch.inf)

    mae = mpot.metric.MAE("energy_mae", Alias.ti, Alias.energy)
    trainer = mpot.Trainer(
        pot,
        criterion,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        strategies=[stagnation],
        metrics=[mae],
        logger={
            "metrics": {
                "step": Identity(Alias.step),
                "epoch": Identity(Alias.epoch),
                "loss": Identity(Alias.loss),
                "energy_mae": MAE(Alias.ti, Alias.energy),
            },
            "handlers": [ConsoleHandler()],
        },
        config={
            "save_dir": "data/lj",
            "device": {"type": "cpu"},
            "report_rate": 10,
            "valid_rate": 1000,
            "modify_lr_rate": 100,
        },
    )
    # trainer.jit()
    trainer.train(10000)
    print("done")
    return "done"

if __name__ == "__main__":
    proj_dir = Path.cwd() / "train_lj"
    proj_dir.mkdir(exist_ok=True)
    os.chdir(proj_dir)
    train_lj(load_lj())
