from enum import IntEnum
from pathlib import Path

import torch
import torch.nn as nn

import molpot as mpot
from molpot.engine.fix import FixManager
from molpot.log import setup_logger

from .base import Engine

from tensordict import TensorDict


class PotentialTrainer(Engine):

    class Stage(IntEnum):

        before_train = 0
        before_epoch = 1
        before_step = 2
        after_step = 3
        after_epoch = 4
        after_train = 5

    def __init__(
        self,
        name: str,
        model: nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        root_dir: str | Path = Path.cwd(),
        enable_amp: bool = False,
        device: str = "cuda"
    ):
        self.name = name
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.root_dir = Path(root_dir)
        self.work_dir = self.root_dir / name
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(name=self.name, output_dir=self.work_dir)
        self.amp_enabled = enable_amp
        self._fix = FixManager(self.Stage)
        # TODO: amp

    def init_status(self) -> TensorDict:
        return TensorDict(
            current_step=0,
            current_epoch=0,
            flag=self.Status.INIT,
            metrices={}
        )
    
    def compile(self) -> None:
        self.model = torch.compile(self.model)

    def train(
        self,
        dataloader,
        steps: int,
        epochs: int = 0,
        upto: bool = False,
        resume: str | Path | bool = False
    ) -> dict:

        step_to_run = steps - 1
        self._fix.register(
            mpot.engine.fix.StepCounter(step_to_run), self.Stage.before_step
        )

        status = {
            "current_step": 0,
            "current_epoch": 0,
            "flag": self.Status.INIT,
            "metrices": {}
        }

        self.before_train(status)
        self._fix.apply(self.Stage.before_train, self, status, None)
        if status['flag'] > self.Status.STOPPING:
            return status

        while True:

            self.before_epoch(status)
            self._fix.apply(self.Stage.before_epoch, self, status, None)
            if status['flag'] > self.Status.STOPPING:
                break

            for data in dataloader:
                data = data.to(self.device)
                self.before_step(status, data)
                self._fix.apply(self.Stage.before_step, self, status, data)
                if status['flag'] > self.Status.STOPPING:
                    break

                self.train_impl(status, data)
                status['current_step'] += 1
                self.after_step(status, data)
                self._fix.apply(self.Stage.after_step, self, status, data)

            self.after_epoch(status, data)
            self._fix.apply(self.Stage.after_epoch, self, status, data)
            status['current_epoch'] += 1

        self.after_train(status, data)  # NOTE: potential bug: this data is from `data = next(iter(dataloader))`
        self._fix.apply(self.Stage.after_train, self, status, data)

        return status

    def before_train(self, status: dict) -> None:
        pass

    def before_epoch(self, status: dict) -> None:
        pass

    def before_step(self, status: dict, inputs: dict) -> None:
        pass

    def train_impl(self, status: dict, inputs: dict) -> None:

        optimizer = self.optimizer
        optimizer.zero_grad()
        lr_scheduler = self.lr_scheduler

        # calculate loss
        inputs = self.model(inputs)
        loss = self.loss_fn(inputs)

        # calculate grad
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        status['metrices']['loss'] = loss.item()

    def after_step(self, status: dict, inputs: dict) -> None:
        pass

    def after_epoch(self, status: dict, inputs: dict) -> None:
        pass

    def after_train(self, status: dict, inputs: dict) -> None:
        pass

    def save_ckpt(self, path: str | Path) -> None:
        pass

    def load_ckpt(self, path: str | Path) -> None:
        pass
