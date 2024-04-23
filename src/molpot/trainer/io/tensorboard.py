import datetime
import time

import torch

from ..fix import Fix


class ConsoloLogFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int=0, **kwargs) -> None:

        super().__init__(every_n_steps, every_n_epochs)
        self.priority = 10

    def before_train(self) -> None:
        self.train_start_time = time.perf_counter()

    def after_train(self) -> None:
        
        total_train_time = time.perf_counter() - self.train_start_time
        
        print(f"Total training time: {total_train_time}")

    def after_iter(self) -> None:
        msg = " ".join([f"{key}: {value:.4f}" for key, value in self.trainer.metrics.items()])
        print(msg)

class TensorBoardFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int, tb_log_dir:str = "tb_log", **kwargs) -> None:

        super().__init__(every_n_steps, every_n_epochs)
        self.tb_log_dir = tb_log_dir
        self.kwargs = kwargs
        self.priority = 10

    def before_train(self) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(self.trainer.work_dir / self.tb_log_dir, **self.kwargs)

    def after_train(self) -> None:
        self._tb_writer.close()

    def _write_tensorboard(self) -> None:
        for key, value in self.trainer.metrics.items():
            self._tb_writer.add_scalar(key, value, self.trainer.steps)


    def after_iter(self) -> None:
        self._write_tensorboard()