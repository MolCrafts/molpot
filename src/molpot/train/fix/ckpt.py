from pathlib import Path

from . import Fix
from collections import deque


class SaveCheckPoint(Fix):

    def __init__(
        self, every_n_steps: int, every_n_epochs: int, max_to_keep: int | None = None
    ) -> None:

        super().__init__(priority=9)

        self.max_to_keep = max_to_keep
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.recent_ckpts = deque(maxlen=max_to_keep)

    def __call__(self, trainer, status, inputs, outputs) -> None:

        step = status["current_step"]
        epoch = status["current_epoch"]
        if step % self.every_n_steps == 0 or epoch % self.every_n_epochs == 0:
            name = f"step_{step}_epoch_{epoch}.pth"
            self.save_ckpt(name, trainer, status, inputs, outputs)

            if len(self.recent_ckpts) == self.max_to_keep:
                to_delete = self.recent_ckpts.popleft()
                self._delete_ckpt(to_delete)

            self.recent_ckpts.append(name)

    def save_ckpt(self, name, train, status, inputs, outputs) -> None:

        print(f"Saving checkpoint {name}")

    def _delete_ckpt(self, name) -> None:

        print(f"Deleting checkpoint {name}")
