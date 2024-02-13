from pathlib import Path
import torch
from molpot.trainer.logger.adapter import LogAdapter
from molpot.trainer.strategy.base import StrategyManager
from molpot.trainer.strategy.early_stop import StepCounter
from molpot.trainer.utils import prepare_device
from ..potentials import NNPotential
import logging
from molpot import alias
import numpy as np
import warnings
import time

class BaseTrainer:
    def __init__(self, name, model: NNPotential, config: dict):
        self.name = name
        self.model = model
        self.config = config

        self.logger = logging.getLogger(self.__class__.__name__)

    def _save_checkpoint(self, step, save_best=False):
        model = type(self.model).__name__
        state = {
            "name": self.name,
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
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    def __init__(
        self,
        name,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        train_data_loader,
        valid_data_loader,
        strategies=[],
        metrics=[],
        logger=None,
        config={},
    ):
        super().__init__(name, model, config)

        self.criterion = criterion
        self.optimizer = optimizer

        self.save_dir = Path(config["save_dir"])

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.metrics = metrics

        self.lr_scheduler = lr_scheduler
        self.device, self.device_ids = prepare_device(config["device"])

        self.strategies = StrategyManager(strategies)

        self.metrics = metrics
        self.log_config = logger
        self.logger_adapter = LogAdapter(name, **logger)

        self.start_step = None
        self.checkpoint_rate = config.get("checkpoint_rate", 1000)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", self.save_dir / "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()

    def jit(self):
        self.model = torch.compile(self.model)
        self.model = self.model.to(self.device)

    def train(self, nsteps: int):

        output = self._pre_train()
        stepCounter = StepCounter(nsteps)
        self.strategies.append(stepCounter)
        nstep = self.start_step
        nepoch = 0
        while True:
            # Training
            self.model.train()
            for data in self.train_data_loader:
                for k, v in data.items():
                    data[k] = v.to(self.device)
                self.optimizer.zero_grad()
                _output = self.model(data)
                loss = self.criterion(_output, data)
                loss.backward()
                self.optimizer.step()
                _output[alias.loss] = loss
                # _output[alias.step] = nstep
                # _output[alias.epoch] = nepoch
                output.update(_output)

                if nstep % self.config["report_rate"] == 0:
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    speed = (nstep - self.start_step) / elapsed_time
                    output['speed'] = speed
                    self.logger_adapter(nstep, nepoch, output, data)

                if nstep % self.config['modify_lr_rate'] == 0:
                    self.lr_scheduler.step()

                if self.strategies(nstep, output, data):
                    if nstep < nsteps:
                        logging.warning(f"Training stopped at step {nstep} due to early stopping.")
                    self._post_train(nstep, output, data)
                    return output

                if nstep % self.config["valid_rate"] == 0:
                    # Validation
                    self.model.eval()
                    with torch.no_grad():
                        for data in self.valid_data_loader:
                            for k, v in data.items():
                                data[k] = v.to(self.device)
                            _output = self.model(data)
                            _output = self.criterion(_output, data)
                    self.model.train()

                if nstep % self.checkpoint_rate == 0:
                    self._save_checkpoint(nstep)
                        
                nstep += 1
            
            nepoch += 1

    def _pre_train(self):
        if self.start_step is None:
            self.start_step = 0
        self.logger_adapter.init()
        return {}

    def _post_train(self, nstep: int, output: dict, data: dict):
        return output

    def eval(self):
        pass

    def load(self, path: str):
        self._resume_checkpoint(path)

class OfflineALTrainer(Trainer):
    def _post_iter(self, nstep: int, output: dict, data: dict):
        return super()._post_iter(output)


class OnlineALTrainer(Trainer):
    pass
