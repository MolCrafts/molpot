import datetime
import time

import torch

from ..fix import Fix


class ConsoloLogFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int=0, separator: str= " | ", **kwargs) -> None:

        super().__init__(every_n_steps, every_n_epochs)
        self.priority = 10
        self.separator = separator

    def before_train(self) -> None:
        keys = ["  steps "] + [f"{key:^12}" for key in self.trainer.metrics.keys()]
        if hasattr(self.trainer, "valid_results"):
            keys += ["{:^12}".format(f"valid_{key}") for key in self.trainer.metrics.keys()]
        print(f"Start training at {datetime.datetime.now()}")
        print(self.separator.join(keys))
        self.train_start_time = time.perf_counter()

    def after_train(self) -> None:
        
        total_train_time = time.perf_counter() - self.train_start_time
        
        print(f"Total training time: {total_train_time:.2f} seconds")

    def after_iter(self) -> None:
        values = [f"{self.trainer.steps:>8}"] + [f"{v:>12.2f}" for v in self.trainer.metrics.values()]
        if hasattr(self.trainer, "valid_results"):
            values += [f"{v:>12.2f}" for v in self.trainer.valid_results.values()]

        msg = self.separator.join(values)
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
        if hasattr(self.trainer, "valid_results"):
            for key, value in self.trainer.valid_results.items():
                self._tb_writer.add_scalar(key, value, self.trainer.steps)


    def after_iter(self) -> None:
        self._write_tensorboard()