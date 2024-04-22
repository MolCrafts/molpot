from pathlib import Path

from ..fix import Fix


class CheckPointFix(Fix):

    def __init__(self, every_n_steps:int, every_n_epochs:int, max_to_keep:int|None=None) -> None:

        super().__init__(every_n_steps, every_n_epochs)

        self.max_to_keep = max_to_keep
        self.recent_ckpts = list()

    def after_iter(self) -> None:
        
        step = self.trainer.steps
        ckpt_name = f"step_{step}.pth"
        self.trainer.save_checkpoint(ckpt_name)
        self.recent_ckpts.append(ckpt_name)
        self._delete_old_ckpts()

    def after_epoch(self) -> None:
        
        epoch = self.trainer.elasped_epochs
        ckpt_name = f"epoch_{epoch}.pth"
        self.trainer.save_checkpoint(ckpt_name)
        self.recent_ckpts.append(ckpt_name)
        self._delete_old_ckpts()

    def _delete_old_ckpts(self):

        if len(self.recent_ckpts) > self.max_to_keep:
            ckpt_to_delete = self.recent_ckpts.pop(0)
            ckpt_path = self.trainer.ckpt_dir / Path(ckpt_to_delete)
            ckpt_path.unlink()
            self.trainer.logger.info(f"Checkpoint {ckpt_to_delete} has been deleted.")

