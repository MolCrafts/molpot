from collections import deque
from pathlib import Path

import torch

from . import Fix


class SaveCheckPoint(Fix):

    def __init__(
        self, every_n_steps: int, every_n_epochs: int, max_to_keep: int | None = None
    ) -> None:

        super().__init__()

        self.max_to_keep = max_to_keep
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.recent_ckpts = deque(maxlen=max_to_keep)

    def __call__(self, trainer, status, inputs) -> None:

        step = status["current_step"]
        epoch = status["current_epoch"]
        if step % self.every_n_steps == 0 or epoch % self.every_n_epochs == 0:
            name = f"step_{step}_epoch_{epoch}.pth"
            self.save_ckpt(name, trainer, status, inputs)

            if len(self.recent_ckpts) == self.max_to_keep:
                to_delete = self.recent_ckpts.popleft()
                self._delete_ckpt(to_delete)

            self.recent_ckpts.append(name)

    def save_ckpt(self, name, train, status, inputs) -> None:

        data = {
            "model": train.model.state_dict(),
            "optimizer": train.optimizer.state_dict(),
            "lr_scheduler": train.lr_scheduler.state_dict(),
            "status": status,
            "fix": train.fixes.state_dict(),
        }
        torch.save(data, Path(train.work_dir) / name)

    def _delete_ckpt(self, name) -> None:

        path = Path(self.work_dir) / name
        if path.exists():
            path.unlink()


class LoadCheckPoint(Fix):

    def __init__(self, every_n_steps: int, every_n_epochs: int, name:str|None = None) -> None:

        super().__init__()

        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.name = name

    def __call__(self, trainer, status, inputs) -> None:

        step = status["current_step"]
        epoch = status["current_epoch"]
        if step % self.every_n_steps == 0 or epoch % self.every_n_epochs == 0:
            self.load_ckpt(trainer, status, inputs)

    def load_ckpt(self, train, status, inputs) -> None:

        if self.name is None:
            ckpt = Path(train.work_dir) / "latest.pth"

        else:
            ckpt = Path(train.work_dir) / f"step_{status['current_step']}_epoch_{status['current_epoch']}.pth"

        assert ckpt.exists(), FileNotFoundError(f"Checkpoint {ckpt} not found.")
        
        if ckpt is not None:
            data = torch.load(ckpt)
            train.model.load_state_dict(data["model"])
            train.optimizer.load_state_dict(data["optimizer"])
            train.lr_scheduler.load_state_dict(data["lr_scheduler"])
            status.update(data["status"])
            train.fixes.load_state_dict(data["fix"])