# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-12
# version: 0.0.1

from torch import nn

__all__ = ["MultiMSELoss"]

class MultiMSELoss(nn.Module):

    def __init__(self, multipliers, targets):
        super().__init__()
        self.multipliers = multipliers
        self.loss_kernel = nn.MSELoss()
        self.targets = targets
        assert len(multipliers) == len(targets)

    def forward(self, outputs):
        loss = 0
        for m, (output_key, label_key) in zip(self.multipliers, self.targets):
            loss += m * self.loss_kernel(outputs[output_key], outputs[label_key])
        return loss