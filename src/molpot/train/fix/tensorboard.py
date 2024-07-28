import datetime
import time
from .base import Fix


class ConsoloLogFix(Fix):
    pass


class TensorBoardFix(Fix):

    def __init__(
        self,
        every_n_steps: int,
        every_n_epochs: int = 1,
        log_dir: str = "tb_log",
        outputs=[],
        **kwargs,
    ) -> None:

        super().__init__(priority=9)
        self.log_dir = log_dir
        self.kwargs = kwargs
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.outputs = outputs
        print(self.outputs)
        from torch.utils.tensorboard import SummaryWriter

        self._tb_writer = SummaryWriter(log_dir, **self.kwargs)

    def __call__(self, trainer, status, inputs, outputs) -> None:

        step = status["current_step"]

        if step % self.every_n_steps == 0:

            for key in self.outputs:
                if key in status['metrices']:
                    self._write_scalar(step, key, status['metrices'][key])
                else:
                    Warning(f"Key {key} not found in status['metrices']")

    def __del__(self) -> None:
        writer = getattr(self, "_tb_writer", None)
        if writer is not None:
            self._tb_writer.close()

    def _write_scalar(self, step: int, key, value, tag='metrices') -> None:
        self._tb_writer.add_scalar(f'{tag}/{key}', value, step, new_style=True)
