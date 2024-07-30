from enum import IntEnum
from pathlib import Path

import torch
import torch.nn as nn

import molpot as mpot
from molpot.engine.fix import FixManager
from molpot.log import setup_logger

from .base import Engine


class PotentialTrainer(Engine):

    class Stage(IntEnum):

        before_train = 0
        before_epoch = 1
        before_iter = 2
        after_iter = 3
        after_epoch = 4
        after_train = 5

    def __init__(
        self,
        name: str,
        model: nn.Module,
        train_dataloader: mpot.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        root_dir: str | Path = Path.cwd(),
        enable_amp: bool = False,
        valid_dataloader: mpot.DataLoader | None = None,
    ):
        self.name = name
        self.model = model
        self.dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
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

    def train(
        self,
        steps: int,
        epochs: int = 0,
        upto: bool = False,
        resume: str | Path | bool = False
    ) -> dict:

        step_to_run = steps
        self._fix.register(
            self.Stage.before_iter, mpot.engine.fix.StepCounter(step_to_run)
        )
        if self.valid_dataloader is not None:
            self._fix.register(
                self.Stage.after_epoch, mpot.engine.fix.Validation(every_n_epoch=1)
            )

        status = {
            "current_step": 0,
            "current_epoch": 0,
            "flag": self.Status.INIT,
            "metrices": {}
        }

        inputs = {}
        outputs = {
            'loss_list': [],
        }

        self.before_train(status, inputs, outputs)
        self._fix.apply(self.Stage.before_train, self, status, inputs, outputs)
        if status['flag'] > self.Status.STOPPING:
            return status, outputs

        while True:

            self.before_epoch(status, inputs, outputs)
            self._fix.apply(self.Stage.before_epoch, self, status, inputs, outputs)
            if status['flag'] > self.Status.STOPPING:
                break

            for data in self.dataloader:

                inputs |= data

                self.before_iter(status, inputs, outputs)
                self._fix.apply(self.Stage.before_iter, self, status, inputs, outputs)
                if status['flag'] > self.Status.STOPPING:
                    break

                self.train_impl(status, inputs, outputs)
                status['current_step'] += 1
                self.after_iter(status, inputs, outputs)
                self._fix.apply(self.Stage.after_iter, self, status, inputs, outputs)

            self.after_epoch(status, inputs, outputs)
            self._fix.apply(self.Stage.after_epoch, self, status, inputs, outputs)
            status['current_epoch'] += 1

        self.after_train(status, inputs, outputs)
        self._fix.apply(self.Stage.after_iter, self, status, inputs, outputs)

        return status, outputs

    def before_train(self, status: dict, inputs: dict, outputs: dict) -> None:
        pass

    def before_epoch(self, status: dict, inputs: dict, outputs: dict) -> None:
        pass

    def before_iter(self, status: dict, inputs: dict, outputs: dict) -> None:
        pass

    def train_impl(self, status: dict, inputs: dict, outputs: dict) -> None:
        model = self.model.train()
        optimizer = self.optimizer
        optimizer.zero_grad()
        lr_scheduler = self.lr_scheduler

        # calculate loss
        inputs, outputs = model(inputs, outputs)
        loss = self.loss_fn(inputs, outputs)

        # calculate grad
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        status['metrices']['loss'] = loss.item()

    def after_iter(self, status: dict, inputs: dict, outputs: dict) -> None:
        pass

    def after_epoch(self, status: dict, inputs: dict, outputs: dict) -> None:
        pass

    def after_train(self, status: dict, inputs: dict, outputs: dict) -> None:
        pass

    def save_ckpt(self, path: str | Path) -> None:
        pass

    def load_ckpt(self, path: str | Path) -> None:
        pass
