import molpot as mpot
from .base import MolPotApp
from .utils import import_from
import torch
from typing import Literal
from torch.optim.lr_scheduler import ExponentialLR
from ignite.engine import Events
from ignite.handlers import global_step_from_engine


class PiNetPotential(MolPotApp):

    def __init__(
        self,
        depth: int = 5,
        r_cutoff: float = 5.0,
        n_basis: int = 10,
        pi_nodes: list[int] = [64, 64],
        ii_nodes: list[int] = [64, 64, 64, 64],
        pp_nodes: list[int] = [64, 64, 64, 64],
        out_nodes: list[int] = [64, 1],
        activation: Literal["tanh"] = "tanh",
        rank=1,
    ):

        if activation == "tanh":
            activation = torch.nn.Tanh()

        pinet = mpot.potential.nnp.PiNet(
            depth=depth,
            basis_fn=mpot.potential.nnp.radial.GaussianRBF(n_basis, r_cutoff),
            cutoff_fn=mpot.potential.nnp.cutoff.CosineCutoff(r_cutoff),
            pi_nodes=pi_nodes,
            ii_nodes=ii_nodes,
            pp_nodes=pp_nodes,
            activation=activation,
            rank=rank,
        )
        e_readout = mpot.potential.nnp.readout.Atomwise(
            out_nodes, from_key=("pinet", "p1"), to_key=("predicts", "energy")
        )
        f_readout = mpot.potential.nnp.readout.DPairPot(
            fx_key=("predicts", "energy"),
            dx_key=("pairs", "dist"),
            to_key=("predicts", "forces"),
            create_graph=True,
        )
        self.model = mpot.potential.PotentialSeq("pinet", pinet, e_readout, f_readout)


class DataLoaderApp(MolPotApp):

    def __init__(
        self,
        class_name: str,
        **kwargs,
    ):

        dataset_class = import_from(class_name)
        dataset = dataset_class(**kwargs)

        dataset.add_process(mpot.process.NeighborList(cutoff=5.0))

        train_ds, valid_ds = torch.utils.data.random_split(dataset, [0.95, 0.05])

        self.train_dl = mpot.DataLoader(
            dataset=train_ds,
            batch_size=10,
            shuffle=False,
        )
        self.eval_dl = mpot.DataLoader(
            dataset=valid_ds,
            batch_size=10,
            shuffle=False,
        )


class PotentialTrainerApp(MolPotApp):

    def __init__(
        self,
        model_app,
        dataloader_app,
        e_loss_weight: float = 1.0,
        f_loss_weight: float = 1.0,
        lr: float = 1e-4
    ): 
        self.e_loss_weight = e_loss_weight
        self.f_loss_weight = f_loss_weight
        self.lr = lr
        self.model_app = model_app
        self.dataset_app = dataloader_app
        
    def init_trainer(self):
        # self.n_instances = n_instances

        loss_keys = []
        if self.e_loss_weight > 0:
            loss_keys.append(("energy", "energy", self.e_loss_weight))
        if self.f_loss_weight > 0:
            loss_keys.append(("forces", "forces", self.f_loss_weight))

        loss_fn = mpot.loss.MultiTargetLoss(torch.nn.MSELoss(), loss_keys)
        model = self.model_app.model

        self.trainer = trainer = mpot.PotentialTrainer(
            model,
            optimizer=torch.optim.Adam(model.parameters(), lr=lr),
            loss_fn=loss_fn,
            device="cuda",
        )
        trainer.add_lr_scheduler(ExponentialLR, gamma=0.994)
        trainer.add_checkpoint(f"pinet/rmd17/ckpt", n_every=int(1e6))
        tb_logger = trainer.attach_tensorboard(
            log_dir="pinet/rmd17/tb_logs",
            event_name=Events.ITERATION_COMPLETED(every=100),
        )

        tb_logger.attach_output_handler(
            trainer.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="trainer",
            output_transform=lambda x: {"loss": x[-1]},
            global_step_transform=global_step_from_engine(trainer.trainer),
        )

        tb_logger.attach_output_handler(
            trainer.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="trainer",
            metric_names=["e_mae", "f_mae"],
            global_step_transform=global_step_from_engine(trainer.trainer),
        )

        tb_logger.attach_output_handler(
            trainer.evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="evaluator",
            metric_names=["e_mae", "f_mae"],
            global_step_transform=global_step_from_engine(trainer.trainer),
        )

    def run(
        self,
        max_steps: int | None = None,
        max_epochs: int | None = None,
        epoch_length: int | None = None,
    ):
        self.trainer.run(
            self.dataloader_app.train_dl,
            max_steps=max_steps,
            max_epochs=max_epochs,
            eval_data=self.dataloader_app.eval_dl,
            epoch_length=epoch_length,
        )

params = {

    "model":
      {"name": "potential_model",
      "params":
        {"e_loss_multiplier": 1.0,
        "f_loss_multiplier": 10.0,
        "log_e_per_atom": True,
        "use_e_per_atom": False,
        "use_force": True,}},
    "network":
      {"name": "PiNet2",
      "params":
        {"atom_types": [1, 6, 7, 8],
        "basis_type": "gaussian",
        "depth": 5,
        "n_basis": 10,
        "pi_nodes": [64],
        "ii_nodes": [64, 64, 64, 64],
        "pp_nodes": [64, 64, 64, 64],
        "out_nodes": [64],
        "rank": 3,
        "rc": 4.5}
       },
    "optimizer":
      {"class_name": "Adam",
      "config":
        {"global_clipnorm": 0.01,
        "learning_rate":
          {"class_name": "ExponentialDecay",
          "config":
            {"decay_rate": 0.994,
            "decay_steps": 100000,
            "initial_learning_rate": 1e-4,}}}}

}
