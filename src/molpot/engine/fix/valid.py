from .base import Fix


class Validation(Fix):

    def __init__(self, every_n_epoch: int, valid_dataloader):
        super().__init__()
        self.every_n_epoch = every_n_epoch
        self.valid_dataloader = valid_dataloader

    def __call__(self, trainer, status, inputs):

        model = trainer.model
        loss_fn = trainer.loss_fn
        for inputs in self.valid_dataloader:
            inputs = inputs.to(trainer.device)
            inputs = model(inputs)
            loss = loss_fn(inputs)
            status['metrices']["valid_loss"] = loss.item()
