import torch
import torch.nn as nn


class Constraint(nn.Module):

    def __init__(self, loss_kernel: nn.Module):
        super().__init__()
        self.loss_kernel = loss_kernel
        self.constraints = []

    def add_error(self, name: str, target, label, weight=1.0, ):
        self.constraints.append((name, target, label, weight))

    def forward(self, pred, label):
        return torch.sum(torch.stack([
            weight * self.loss_kernel(pred[target_key], label[label_key])
            for _, target_key, label_key, weight in self.constraints
        ]))