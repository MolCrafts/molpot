from pathlib import Path
from pprint import pprint
import torch
from molpot.trainer.logger.adapter import LogAdapter
from molpot.trainer.strategy.base import StrategyManager
from molpot.trainer.strategy.early_stop import StepCounter
from ..potentials import Potentials
import logging
from molpot import Alias, Config
import time
from torch.export import export

class BaseTrainer:
    def __init__(self, name, model: Potentials, config: dict):
        self.name = name
        self.model = model
        self.config = config

        self.logger = logging.getLogger(self.__class__.__name__)

    def save_model(self, fpath, train_state: dict):
        model = self.model.__class__.__name__
        state = {
            "name": self.name,
            "model": {"name": model, "state_dict": self.model.state_dict()},
            "train_state": train_state,
        }
        torch.save(state, fpath)

    def load_model(self, resume_path):
        resume_path = Path(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path.absolute()))
        if not resume_path.exists():
            raise FileNotFoundError(
                "Checkpoint file not found: {}".format(resume_path.absolute())
            )
        state = torch.load(resume_path)
        train_state = state["train_state"]
        model = state["model"]
        self.start_step = train_state["step"] + 1
        self.start_epoch = train_state["epoch"] + 1

        # load architecture params from checkpoint.
        if model["name"] != self.model.__class__.__name__:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(model["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        # if train_state["optimizer"]["type"] != self.config["optimizer"]["type"]:
        #     self.logger.warning(
        #         "Warning: Optimizer type given in config file is different from that of checkpoint. "
        #         "Optimizer parameters not being resumed."
        #     )
        # else:
        #     self.optimizer.load_state_dict(train_state["optimizer"])
        self.optimizer.load_state_dict(train_state["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )


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
        logger=None,
        config={},
    ):
        super().__init__(name, model, config)

        self.criterion = criterion
        self.optimizer = optimizer

        self.save_dir = Path(config["save_dir"])

        if config.get("compile", False):
            self.logger.info("Compiling model...")
            self.model = torch.compile(self.model)
            self.model = self.model.to(Config.device)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.lr_scheduler = lr_scheduler
        Config.set_device(config["device"])

        self.strategies = StrategyManager(strategies)

        self.log_config = logger
        self.logger_adapter = LogAdapter(name, **self.log_config)

        self.start_step = None
        self.checkpoint_rate = config.get("checkpoint_rate", 1000)

        self.start_time = time.time()
        resume = config.get("resume", None)

        if resume:
            self.load_model(config["resume"])
        else:
            self.checkpoint_dir = Path(
                config.get("checkpoint_dir", self.save_dir / "checkpoints")
            )
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


    def train(self, nsteps: int):

        output = self._pre_train()
        stepCounter = StepCounter(nsteps)
        self.strategies.append(stepCounter)
        nstep = self.start_step
        nepoch = self.start_epoch
        while True:
            # Training
            self.model.train()
            for data in self.train_data_loader:
                for k, v in data.items():
                    data[k] = v.to(Config.device)
                self.optimizer.zero_grad()
                _output = self.model(data)
                loss = self.criterion(_output)
                loss.backward()
                self.optimizer.step()
                _output[Alias.loss] = loss
                output.update(_output)

                if nstep % self.config["report_rate"] == 0:
                    current_time = time.time()
                    elapsed_time = current_time - self.start_time
                    speed = (nstep - self.start_step) / elapsed_time
                    output["speed"] = speed
                    self.logger_adapter(nstep, nepoch, output, data)

                if nstep % self.config["modify_lr_rate"] == 0:
                    self.lr_scheduler.step()

                if self.strategies(nstep, output, data):
                    if nstep < nsteps:
                        self.logger.warning(
                            f"Training stopped at step {nstep} due to early stopping."
                        )
                    self._post_train()
                    return output

                if nstep % self.config["valid_rate"] == 0:
                    # Validation
                    self.model.eval()
                    with torch.no_grad():
                        for data in self.valid_data_loader:
                            for k, v in data.items():
                                data[k] = v.to(Config.device)
                            _output = self.model(data)
                            _output = self.criterion(_output, data)
                    self.model.train()

                if nstep % self.checkpoint_rate == 0:
                    checkpoint_name = self.checkpoint_dir / f"{self.name}-{nstep}.pt"
                    self.train_state["step"] = nstep
                    self.train_state["epoch"] = nepoch
                    self.save_model(checkpoint_name, self.train_state)

                nstep += 1

            nepoch += 1

    def _pre_train(self):
        if self.start_step is None:
            self.start_step = 0
            self.start_epoch = 0
        self.logger_adapter.init()
        self.train_state = {
            "step": self.start_step,
            "epoch": self.start_epoch,
            "finish": False,
            "optimizer": self.optimizer.state_dict(),
        }
        return {}

    def _post_train(self):
        final_model = self.save_dir / f"{self.name}.pt"
        self.train_state["finish"] = True
        self.save_model(final_model, self.train_state)
        return {}
    
    def export(self):
        for data in self.train_data_loader:
            # TODO: export factory
            pass

class OfflineALTrainer(Trainer):
    def _post_iter(self, nstep: int, output: dict, data: dict):
        return super()._post_iter(output)


class OnlineALTrainer(Trainer):
    pass
