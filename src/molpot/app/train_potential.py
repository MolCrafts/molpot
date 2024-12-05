import molpot as mpot
from .config import ConfigProcessor
from pydantic import BaseModel
from ignite.metrics import MeanAbsoluteError, BatchWise
from ignite.handlers import global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine.events import Events
from pathlib import Path

class App(BaseModel):
    ...


class TrainPotential(App):

    def __init__(self, model, optimizer, loss_fn, use_energy, use_forces, device):
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            use_energy=use_energy,
            use_forces=use_forces,
            device=device
        )

    def process(self):

        self.trainer = mpot.PotentialTrainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device
        )

        if self.use_energy:
            def get_energy(data):
                return data[0]["energy"], data[1]["energy"]
            self.trainer.add_metric("e_mae", MeanAbsoluteError(output_transform=get_energy),
    target="all",
    usage=BatchWise())
            
        if self.use_forces:
            def get_forces(data):
                return data[0]["forces"], data[1]["forces"]
            self.trainer.add_metric(
            "f_mae",
            MeanAbsoluteError(output_transform=get_forces),
            target="all",
            usage=BatchWise(),
        )
            
        self.trainer.attach_progressbar()


        tb_logger = TensorboardLogger(Path("rmd17_log"))
        tb_logger.attach_output_handler(
            self.trainer.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="trainer",
            output_transform=lambda x: {"loss": x[-1]},
            global_step_transform=global_step_from_engine(self.trainer.trainer),
        )

        metric_name = []
        if self.use_energy:
            metric_name.append("e_mae")
        if self.use_forces:
            metric_name.append("f_mae")
            

        tb_logger.attach_output_handler(
            self.trainer.trainer,
            event_name=Events.ITERATION_COMPLETED,
            tag="trainer",
            metric_names=["e_mae", "f_mae"],
            global_step_transform=global_step_from_engine(self.trainer.trainer),
        )

        tb_logger.attach_output_handler(
            self.trainer.evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag="evaluator",
            metric_names=["e_mae", "f_mae"],
            global_step_transform=global_step_from_engine(self.trainer.trainer),
        )



    @classmethod
    def from_config(cls, config):
        parser = ConfigProcessor(config)
        ins = cls(
            model=parser.model,
            optimizer=parser.optimizer,
            loss_fn=parser.loss_fn,
            use_energy=parser.use_energy,
            use_forces=parser.use_forces,
            device=parser.device
        )
        return ins

    def run(self, train_dl, max_epochs, eval_dl=None):
        if eval_dl is not None:
            self.trainer.add_evaluator(eval_dl, no_grad=False)
        return self.trainer.run(train_dl, max_epochs)