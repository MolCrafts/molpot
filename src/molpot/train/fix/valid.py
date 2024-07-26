from .base import Fix

class Validation(Fix):

    def __init__(self, every_n_epoch:int):
        super().__init__(priority=1)
        self.every_n_epoch = every_n_epoch

    def __call__(self, trainer, status, inputs, outputs):
        
        model = trainer.model.eval()
        dataloader = trainer.valid_dataloader
        loss_fn = trainer.loss_fn
        for inputs in dataloader:
            
            inputs, outputs = model(inputs, outputs)
            loss = loss_fn(inputs, outputs)
            outputs['valid_loss'] = loss.item()

        model.train()