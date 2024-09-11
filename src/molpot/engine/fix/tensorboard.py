import datetime
import time
from .base import Fix
from pathlib import Path
import torch

class ConsoloLogFix(Fix):
    pass


class TensorBoardFix(Fix):

    def __init__(
        self,
        every_n_steps: int,
        every_n_epochs: int = 1,
        log_dir: str = "tb_log",
        excludes=[],
        **kwargs,
    ) -> None:

        super().__init__()
        self.log_dir = log_dir
        self.kwargs = kwargs
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.excludes = excludes
        from torch.utils.tensorboard import SummaryWriter

        self._tb_writer = SummaryWriter(log_dir, **self.kwargs)

    def __call__(self, trainer, status, inputs) -> None:

        step = status["current_step"]

        if step % self.every_n_steps == 0:

            for key, value in status["metrices"].items():
                if key not in self.excludes:
                    self._write_scalar(step, key, value)

    def __del__(self) -> None:
        writer = getattr(self, "_tb_writer", None)
        if writer is not None:
            self._tb_writer.close()

    def _write_scalar(self, step: int, key, value, tag='metrices') -> None:
        self._tb_writer.add_scalar(f'{tag}/{key}', value, step, new_style=True)

class Profiler(Fix):

    def __init__(
        self,
        wait, warmup, active, repeat=0, skip_first=0, log_dir=Path.cwd(), record_shapes=True, with_stack=True
    ):
        super().__init__()
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=record_shapes,
            with_stack=with_stack
        )

    def before_epoch(self, trainer, status, inputs):
        self.profiler.start()

    def after_step(self, trainer, status, inputs):
        self.profiler.step()

    def after_epoch(self, trainer, status, inputs):
        trainer._fix.remove(self)
        self.profiler.stop()