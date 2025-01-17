import torch

class MultiTargetLoss(torch.nn.Module):

    def __init__(self, loss_kernel: torch.nn.Module, keys):
        super().__init__()
        self.loss_kernel = loss_kernel
        self.keys = keys

    def forward(self, pred, label):
        loss = 0
        for key, target, weight in self.keys:
            loss += weight * self.loss_kernel(pred[key], label[target])
        return loss
