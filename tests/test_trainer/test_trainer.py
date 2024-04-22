# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-03
# version: 0.0.1
import torch
import molpot as mpot
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from torchdata.dataloader2 import DataLoader2
import molpy as mp


def test_classical_trainer():

    frame = mp.Frame()
    spec, ff = mp.structure.h2o.spc_fw().values()
    frame.add_struct(spec)
    spec, ff = mp.structure.h2o.spc_fw().values()
    frame.add_struct(spec)

    frame.set_box(10, 10, 10)
    frame.calc_neighborlist(2.0)
    frame.calc_connectivity()

    frame1 = frame.clone()
    frame1.atoms.xyz += 1
    frame2 = frame.clone()
    frame2.atoms.xyz += 2

    dp = IterableWrapper([frame, frame1, frame2])
    dl = DataLoader2(dp)

    lj126 = mpot.potential.pair.LJ126(
        epsilon=ff.get_pairstyle("lj/cut/coul/cut").get_pairtype_params("epsilon"),
        sigma=ff.get_pairstyle("lj/cut/coul/cut").get_pairtype_params("sigma"),
        cutoff=2.5,
    )
    # coul = mpot.potential.pair.ewald.Ewald()
    bond = mpot.potential.bond.Harmonic(
        k=ff.get_bondstyle("harmonic").get_bondtype_params("k"),
        r0=ff.get_bondstyle("harmonic").get_bondtype_params("r0"),
    )
    angle = mpot.potential.angle.Harmonic(
        k=ff.get_anglestyle("harmonic").get_angletype_params("k"),
        theta0=ff.get_anglestyle("harmonic").get_angletype_params("theta0"),
    )

    # nnp = mpot.PotentialSeq(mpot.potential.nnp.PaiNN(), mpot.potential.nnp.Atomwise())

    model = mpot.PotentialDict(
        dict(
            lj126=lj126,
            # coul=coul,
            # bond=bond,
            # angle=angle,
        )
        # nnp=nnp,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    work_dir = "./work_dir"

    trainer = mpot.Trainer(model, loss_fn, optimizer, lr_scheduler, dl, work_dir)

    trainer.register_fix(mpot.trainer.strategy.StepCounter())

    trainer.train(10)
