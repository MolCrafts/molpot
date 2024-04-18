from ..fix import Fix


class TensorBoardFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int, tb_log_dir:str = "tb_log", **kwargs) -> None:

        super().__init__(every_n_steps, every_n_epochs)

        from torch.utils.tensorboard import SummaryWriter
        self._tb_writer = SummaryWriter(tb_log_dir, **kwargs)

    def before_train(self) -> None:
        self._train_start_time = time.perf_counter()

    def after_train(self) -> None:
        self._tb_writer.close()
        total_train_time = time.perf_counter() - self._train_start_time
        total_hook_time = total_train_time - self.metric_storage["iter_time"].global_sum
        logger.info("Total training time: {} ({} on hooks)".format(
            str(datetime.timedelta(seconds=int(total_train_time))),
            str(datetime.timedelta(seconds=int(total_hook_time))),
        ))

    def after_epoch(self) -> None:
        # Some hooks maybe generate logs in after_epoch().
        # When LoggerHook is the last hook, calling _write_tensorboard()
        # after each epoch can avoid missing logs.
        self._write_tensorboard()

    def _write_console(self) -> None:
        # These fields ("data_time", "iter_time", "lr", "loss") may does not
        # exist when user overwrites `Trainer.train_one_iter()`
        data_time = (self.metric_storage["data_time"].avg
                     if "data_time" in self.metric_storage else None)
        iter_time = (self.metric_storage["iter_time"].avg
                     if "iter_time" in self.metric_storage else None)
        lr = self.metric_storage["lr"].latest if "lr" in self.metric_storage else None

        if iter_time is not None:
            eta_seconds = iter_time * (self.trainer.max_iters - self.trainer.cur_iter - 1)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        else:
            eta_string = None

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        loss_strings = [
            f"{key}: {his_buf.avg:.4g}" for key, his_buf in self.metric_storage.items()
            if "loss" in key
        ]

        if self.trainer.train_by_epoch:
            process_string = "Epoch: [{}][{}/{}]".format(self.trainer.cur_epoch,
                                                         self.trainer.inner_iter,
                                                         self.trainer.epoch_len - 1)
        else:
            process_string = "Iter: [{}/{}]".format(self.trainer.cur_iter,
                                                    self.trainer.max_iters - 1)

        space = " " * 2
        logger.info("{process}{eta}{losses}{iter_time}{data_time}{lr}{memory}".format(
            process=process_string,
            eta=space + f"ETA: {eta_string}" if eta_string is not None else "",
            losses=space + "  ".join(loss_strings) if loss_strings else "",
            iter_time=space + f"iter_time: {iter_time:.4f}" if iter_time is not None else "",
            data_time=space + f"data_time: {data_time:.4f}  " if data_time is not None else "",
            lr=space + f"lr: {lr:.5g}" if lr is not None else "",
            memory=space + f"max_mem: {max_mem_mb:.0f}M" if max_mem_mb is not None else "",
        ))

    def _write_tensorboard(self) -> None:
        for key, (iter, value) in self.metric_storage.values_maybe_smooth.items():
            if key not in self._last_write or iter > self._last_write[key]:
                self._tb_writer.add_scalar(key, value, iter)
                self._last_write[key] = iter

    def after_iter(self) -> None:
        if self.trainer.train_by_epoch and self.every_n_inner_iters(self._period):
            self._write_console()
            self._write_tensorboard()
        if not self.trainer.train_by_epoch and self.every_n_iters(self._period):
            self._write_console()
            self._write_tensorboard()