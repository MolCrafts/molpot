from pathlib import Path
from molpot.trainer.logger.logger import LogAdapter
import torch

from molpot.trainer.utils import prepare_device
from .metric.tracker import MetricTracker
from .metric import get_metric
from ..potentials import NNPotential
import logging
from .strategy import EarlyStop, PlannedStop
from molpot import alias
import numpy as np

class BaseTrainer:
    def __init__(self, model: NNPotential, config: dict):
        self.model = model
        self.config = config

        self.logger = logging.getLogger(self.__class__.__name__)

    def _save_checkpoint(self, step, save_best=False):
        model = type(self.model).__name__
        state = {
            "model": model,
            "step": step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-{}.pth".format(step))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = Path(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_step = checkpoint["step"] + 1
        # self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        # if checkpoint['config']['arch'] != self.config['arch']:
        #     self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                         "checkpoint. This may yield an exception while state_dict is being loaded.")
        # self.model.load_state_dict(checkpoint['state_dict'])

        # # load optimizer state from checkpoint only when optimizer type is not changed.
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        # else:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])

        # self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_data_loader,
        lr_scheduler=None,
        valid_data_loader=None,
        config={},
    ):
        super().__init__(model, config)

        self.criterion = criterion
        self.optimizer = optimizer

        self.save_dir = Path(config["save_dir"])
        self.resume = config.get("resume", None)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.metrics = config["metrics"]
        self.metric_fns = {metric: get_metric(metric) for metric in self.metrics}

        self.lr_scheduler = lr_scheduler
        self.device, self.device_ids = prepare_device(config["device"])

    def train(self, nsteps: int):
        plannedStop = PlannedStop(nsteps)
        earlyStop = EarlyStop()
        output = self._pre_train()
        for i, data in enumerate(self.train_data_loader, self.start_step+1):
            output = self._pre_iter(i, data, output)
            output = self._train(i, data, output)
            output = self._valid(i, data, output)
            output = self._log(i, data, output)
            output = self._post_iter(i, data, output)

            if plannedStop(output["nstep"]) or earlyStop(output['loss']):
                break

        self._post_train(output)

    def _pre_train(self):
        if self.resume:
            self._resume_checkpoint(self.resume)
        else:
            self.start_step = 0

        self.train_metrics = MetricTracker("train")
        self.valid_metrics = MetricTracker("valid")
        self.logger = LogAdapter(self.model.name, self.save_dir)
        return {}

    def _pre_iter(self, nstep, data, output: dict):
        return output

    def _train(self, nstep, data, output: dict):
        self.model.train()
        self.optimizer.zero_grad()
        _output = self.model(data)
        loss = self.criterion(_output, data)
        loss.backward()
        output.update(
            {
                "train": _output,
                "loss": loss,
            }
        )
        self.optimizer.step()
        return output

    def _valid(self, nstep, data, output: dict):

        if nstep % 100 != 0:
            return output

        self.model.eval()

        with torch.no_grad():
            for i, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.valid_metrics.update("loss", loss.item())
                for m, metric_fn in self.metric_fns.items():
                    self.valid_metrics.update(
                        m, metric_fn(output["output"], output["loss"])
                    )
        return output

    def _log(self, nstep, data, output: dict):
        self.train_metrics.update("loss", output["loss"].item())
        for m, metric_fn in self.metric_fns.items():
            self.train_metrics.update(m, metric_fn(output["output"], output["loss"]))
        return output

    def _post_iter(self, nstep, data, output: dict):
        
        if output["nstep"] % 100 == 0:
            self.lr_scheduler.step()

        # print status
        # TODO: use a log adapter

        self.logger.info(
            f"Step {output['nstep']:06d} | Train Loss {self.train_metrics.output['loss']:.4f} | Valid Loss {self.valid_metrics.output.get('loss', np.nan):.4f}"
        )
        return output

    def _post_train(self, nstep, data, output: dict):
        # self._save_checkpoint(self.start_step, save_best=True)
        return output


class OfflineALTrainer(Trainer):
    
    def _post_iter(self, nstep, data, output: dict):
        return super()._post_iter(output)

class OnlineALTrainer(Trainer):
    pass