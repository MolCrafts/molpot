import molpot as mpot
import molpy as mp
import torch
import molexp as me

from molpot.trainer.metric.metrics import Identity, MAE
from molpot.trainer.logger.adapter import ConsoleHandler
from molpot import alias

def gen_lj():

    script = me.Script('lj.in')
    script.content = f"""
        units lj
        dimension 3
        atom_style atomic
        pair_style lj/cut 2.5
        boundary p p p
        region simulation_box block -20 20 -20 20 -20 20
        create_box 1 simulation_box
        create_atoms 1 random 1500 341341 simulation_box
        # create_atoms 2 random 100 127569 simulation_box
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
        run 10000
    """
    script.write()

    me.run("lmp -in lj.in")
    return


def load_lj():

    traj = mp.Trajectory()
    log = mp.loadtxt("lj.log")
    traj.join_frames(log['etotal'])

    lj_dataset = mpot.DataSet.from_traj(traj)
    dp = lj_dataset.prepare()
    train, valid = (
        dp.calc_nblist(5)
        .random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
    )

    train_dataloader = mpot.create_dataloader(train)
    valid_dataloader = mpot.create_dataloader(valid)
    return train_dataloader, valid_dataloader

def train_lj(load_lj) -> str:

    train_dataloader, valid_dataloader = load_lj
    lj_pot = mpot.classical.LJ126(5)
    pot = mpot.Potentials("LJ", lj_pot)
    criterion = mpot.MultiMSELoss([1], targets=[(alias.ti, alias.)])
    optimizer = torch.optim.Adam(pot.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    stagnation = mpot.strategy.Stagnation(alias.loss, patience=torch.inf)

    mae = mpot.metric.MAE("energy_mae", alias.ti, alias.QM9.U)
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
                "step": Identity(alias.step),
                "epoch": Identity(alias.epoch),
                "loss": Identity(alias.loss),
                "energy_mae": MAE(alias.ti, alias.QM9.U),
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
    return "done"

if __name__ == "__main__":
    train_lj(load_lj())