import datetime
import time

import torch

from ..fix import Fix


class ConsoloLogFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int=0, **kwargs) -> None:

        super().__init__(every_n_steps, every_n_epochs)

    def before_train(self) -> None:
        self.train_start_time = time.perf_counter()

    def after_train(self) -> None:
        
        total_train_time = time.perf_counter() - self.train_start_time
        
        print(f"Total training time: {total_train_time}")

    def before_iter(self) -> None:
        self.iter_start_time = time.perf_counter()

    def after_iter(self) -> None:
        self.iter_time = (time.perf_counter() - self.iter_start_time) / self.every_n_steps
        print(f"Step: {self.trainer.steps} Epoch: {self.trainer.epochs} Speed: {self.iter_time}")

class TensorBoardFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int, tb_log_dir:str = "tb_log", **kwargs) -> None:

        super().__init__(every_n_steps, every_n_epochs)
        self.tb_log_dir = tb_log_dir
        self.kwargs = kwargs

    def before_train(self) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(self.trainer.work_dir / self.tb_log_dir, **self.kwargs)

    def after_train(self) -> None:
        self._tb_writer.close()

    def _write_tensorboard(self) -> None:
        for key, value in self.trainer.metrics.items():
            # if key not in self._last_write or iter > self._last_write[key]:
            self._tb_writer.add_scalar(key, value, self.trainer.steps)
            print(key, value, self.trainer.steps)
                # self._last_write[key] = iter

    def after_iter(self) -> None:
        self._write_tensorboard()