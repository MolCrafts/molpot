import weakref
from collections import defaultdict
from enum import Flag, auto
from pathlib import Path

import torch
from torch.cuda.amp import autocast

import molpot as mpot

from .distributed import get_rank, get_world_size
from .fix import FixManager, Fix
from .log import setup_logger


class Trainer:

    class Status(Flag):
        INIT = auto()
        TRAINING = auto()
        STOP_EPOCH = auto()
        STOP_TRAIN = auto()
        VALIDATING = auto()
        FINISHED = auto()
        STOPPED = auto()
        
    def __init__(
        self,
        model: mpot.Potential,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        data_loader: mpot.DataLoader,
        work_dir: str | Path,
        enable_amp: bool = False,
    ):

        self.logger = setup_logger(name="trainer", output_dir=work_dir, rank=get_rank())
        self.trainer_version = "0.1.0"

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.work_dir = Path(work_dir)
        self.data_loader = data_loader

        self.amp_enabled = enable_amp
        self.fixes = FixManager()


        self.status = Trainer.Status.INIT

        self.logger.info(mpot.Config.get_environ())

        self._grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        self.ckpt_dir = self.work_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}

    def train(
        self,
        steps: int,
        epochs: int = 0,
        upto: bool = False,
    ):

        model = self.model
        model.train()

        self._apply_fix("before_train")
        try:
            length_of_data_loader = len(self.data_loader)
            total_steps = steps + epochs * length_of_data_loader
        except TypeError:
            total_steps = steps

        if upto:
            self.train_steps = total_steps - self.steps
            self.train_epochs = epochs - self.epochs
        else:
            self.train_steps = total_steps
            self.train_epochs = epochs

        self.elasped_epochs = 0
        self.elasped_steps = 0
        while True:
            self._apply_fix("before_epoch")
            for self.train_data in self.data_loader:
                self._apply_fix("before_iter")
                self.train_impl(self.train_data)
                self._apply_fix("after_iter")
                self.elasped_steps += 1
                if self.status == Trainer.Status.STOP_EPOCH:
                    break

            if self.status == Trainer.Status.STOP_TRAIN:
                break
            self.elasped_epochs += 1
            self._apply_fix("after_epoch")

        self._apply_fix("after_train")

    def train_impl(self, data):

        with autocast(enabled=self.amp_enabled):

            outputs = self.model(data)
            loss = self.loss_fn(outputs, data)

        self.optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()

        self._grad_scaler.step(self.optimizer)
        self._grad_scaler.update()

        self.train_outputs = outputs
        self.train_loss = loss

    def _apply_fix(self, stage: str):
        for fix in self.fixes:
            getattr(fix, stage)()

    def register_fix(self, fix: Fix):
        fix.trainer = weakref.proxy(self)
        self.fixes.append(fix)


    def save_checkpoint(self, file_name: str|Path)->None:
        data = {
            "version": self.trainer_version,
            "device": get_world_size(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epochs": self.epochs,
            "steps": self.steps,
            "fixes": {}
        }

        if self.amp_enabled:
            data["grad_scaler"] = self.grad_scaler.state_dict()
        
        for fix in self.fixes:
            if fix.checkpointable:
                data["fixes"][fix.name] = fix.state_dict()

        file_path = self.ckpt_dir / Path(file_name)
        self.logger.info(f"Saving checkpoint to {file_path}")
        torch.save(data, file_path)

    def load_checkpoint(self, file_name: str|Path = Path("latest.pth"))->None:
        """Load the given checkpoint or resume from the latest checkpoint.

        Args:
            path (str): Path to the checkpoint to load. If None, load the latest checkpoint.
        """
        if file_name is None:
            path = self.ckpt_dir / file_name
        else:
            path = Path(file_name)

        if path.exists():
            self.logger.info(f"Loading checkpoint from {path} ...")
            checkpoint = torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        if self.trainer_version != checkpoint.get("version"):
            raise SystemError(f"Checkpoint version mismatch: trainer: {self.trainer_version} vs ckpt: {checkpoint.get('version')}")

        # check if the number of GPUs is consistent with the checkpoint
        num_gpus = get_world_size()
        ckpt_num_gpus = checkpoint["device"]
        assert num_gpus == ckpt_num_gpus, (
            f"You are trying to load a checkpoint trained with {ckpt_num_gpus} GPUs, "
            f"but currently only have {num_gpus} GPUs.")

        # 1. load epoch / iteration
        self.epochs = checkpoint["epoch"]
        self.steps = checkpoint["steps"]

        # 2. load model
        self.model.load_state_dict(checkpoint["model"])

        # # 3. load metric_storage
        # self.metric_storage = checkpoint["metric_storage"]

        # 4. load optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # 5. load lr_scheduler
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        # 6. load grad scaler
        consistent_amp = not (self.amp_enabled ^ ("grad_scaler" in checkpoint))
        assert consistent_amp, "Found inconsistent AMP training setting when loading checkpoint."
        if self.amp_enabled:
            self._grad_scaler.load_state_dict(checkpoint["grad_scaler"])

        # 7. load hooks
        fix_states = checkpoint.get("fixes", {})
        fix_names = [fix.name for fix in self.fixes if fix.checkpointable]
        missing_keys = [name for name in fix_names if name not in fix_states]
        unexpected_keys = [key for key in fix_states if key not in fix_names]
        if missing_keys:
            self.logger.warning(f"Encounter missing keys when loading fix state dict:\n{missing_keys}")
        if unexpected_keys:
            self.logger.warning(
                f"Encounter unexpected keys when loading fix state dict:\n{unexpected_keys}")

        for key, value in fix_states.items():
            for fix in self.fixes:
                if fix.name == key and fix.checkpointable:
                    fix.load_state_dict(value)
                    break