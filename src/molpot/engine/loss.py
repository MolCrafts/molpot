from collections import namedtuple
import torch
import torch.nn as nn

constraint = namedtuple("constraint", ["name", "kernel", "target", "label", "weight"])

class Constraint(nn.Module):

    def __init__(self):
        # TODO: auto register to logger
        super().__init__()
        self._losses = []
        self._metrics = []

    def add_loss(
        self,
        kernel,
        target,
        label,
        weight=1.0,
        name: str | None = None,
        log: bool = False,
    ):
        if name is None:
            name = f"{target}-{label} {kernel.__class__.__name__}"
        self._losses.append(constraint(name, kernel, target, label, weight))

        if log:
            self._metrics.append(name)

    def forward(self, pred, label):
        return torch.sum(
            torch.stack(
                [
                    weight * kernel(pred[target_key], label[label_key])
                    for _, kernel, target_key, label_key, weight in self._losses
                ]
            )
        )
