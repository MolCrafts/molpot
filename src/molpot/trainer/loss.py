# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-12
# version: 0.0.1

from torch import nn

__all__ = ["MultiMSELoss"]

class MultiMSELoss(nn.Module):

    def __init__(self, weights, targets):
        super().__init__()
        self.weights = weights
        self.loss_kernel = nn.MSELoss()
        self.targets = targets
        assert len(weights) == len(targets)

    def forward(self, output, data):
        loss = 0
        for weight, (k1, k2) in zip(self.weights, self.targets):
            loss += weight * self.loss_kernel(output[k1], data[k2])
        return loss