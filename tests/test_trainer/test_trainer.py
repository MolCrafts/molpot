# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-03
# version: 0.0.1

from molpot.piplines.dataloaders import create_dataloader
from molpot.potentials.classical.pair.lj import LJ126
from molpot.trainer import Trainer
import torch
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import IterableWrapper


def lj126(r_ij):
    eps = 2
    sig = 1
    return 4 * eps * ((sig / r_ij) ** 12 - (sig / r_ij) ** 6)


def test_trainer():
    def gen_data():
        r = torch.linspace(0.3, 3.0, 100)
        r.requires_grad_(True)
        source_dp = IterableWrapper(r)
        dp1, dp2 = source_dp.fork(2)
        e_ij = dp1.map(lj126)
        r_ij = dp2
        data = r_ij.zip(e_ij)
        train, valid = data.random_split(weights={"train": 0.8, "valid": 0.2}, seed=42)
        print(train)
        return create_dataloader(train, nworkers=None), create_dataloader(
            valid, nworkers=None
        )

    train_data_loader, valid_data_loader = gen_data()

    model = LJ126(eps=torch.tensor([3.0], requires_grad=True), sig=torch.tensor([2.0], requires_grad=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.MSELoss()
    metrics = []
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        metrics,
        train_data_loader=train_data_loader,
        lr_scheduler=lr_scheduler,
        valid_data_loader=valid_data_loader,
        config={
            "trainer": {
                "device": {"type": "cpu" },
                "save_dir": "_saved",
            }
        },
    )
    trainer.train(1000)
