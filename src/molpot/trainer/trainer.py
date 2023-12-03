from pathlib import Path
from molpot.trainer.logger.logger import LogAdapter
import torch
from .metric.tracker import MetricTracker
from .metric import get_metric
from ..potentials import Potential
import logging


class BaseTrainer:
    def __init__(self, model: Potential, config: dict):
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
        model: Potential,
        criterion,
        optimizer,
        metrics,
        train_data_loader,
        lr_scheduler=None,
        valid_data_loader=None,
        config={},
    ):
        super().__init__(model, config)

        self.criterion = criterion
        self.optimizer = optimizer

        self.save_dir = Path(config["trainer"]["save_dir"])
        self.resume = config["trainer"].get("resume", None)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.metrics = metrics
        self.metric_fns = [get_metric(m) for m in self.metrics]

        self.lr_scheduler = lr_scheduler

    def train(self, nstep: int):
        self._pre_train()
        for i, (data, target) in enumerate(self.train_data_loader):
            nstep = i + self.start_step
            result = {
                'nstep': nstep,
                'data': data.to(self.device),
                'target': target.to(self.device)
            }
            result = self._pre_iter(result)
            result = self._train(result)
            result = self._train_log(result)
            result = self._valid(result)
            result = self._valid_log(result)
            result = self._post_iter(result)

            if i >= nstep:
                break
        self._post_train()

    def _pre_train(self):
        if self.resume:
            self._resume_checkpoint(self.resume)
        else:
            self.start_step = 0

        self.train_metrics = MetricTracker("train")
        self.valid_metrics = MetricTracker("valid")
        self.logger = LogAdapter(self.config["name"], self.save_dir)

    def _pre_iter(self):
        self.optimizer.zero_grad()

    def _train(self, result:dict):
        self.model.train()
        data = result["data"]
        target = result["target"]
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        return {
            'output': output,
            'loss': loss,
        }
    
    def _valid(self, result:dict):
        self.model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                result["valid_loss"].update(loss.item(), data.size(0))
        return result

    def _train_log(self, result:dict):
        self.train_metrics.update("loss", result["loss"].item())
        for metric_fn in self.metric_fns:
            self.train_metrics.update(metric_fn(result["output"], result["target"]))

    def _valid_log(self, result:dict):
        self.valid_metrics.update("loss", result["valid_loss"].item())
        for metric_fn in self.metric_fns:
            self.valid_metrics.update(metric_fn(result["output"], result["target"]))

    def _post_iter(self, result:dict):
        self.optimizer.step()
        self.lr_scheduler.step()

        # print status
        # TODO: use a log adapter
        self.logger.info(
            "Step {:06d} | Train Loss {:.4f} | Valid Loss {:.4f}".format(
                result["nstep"], self.train_metrics.result["loss"], self.valid_metrics.result["loss"]
            )
        )

    def _post_train(self, result:dict):
        self._save_checkpoint(self.start_step, save_best=True)
