from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class constraint:
    name: str
    kernel: nn.Module
    target: str
    label: str
    weight: float


class Constraint(nn.Module):

    def __init__(self):
        # TODO: auto register to logger
        super().__init__()
        self._constraints = {}
        self._metrics = []

    def add(
        self,
        name: str,
        kernel,
        target,
        label,
        weight=1.0,
        log: bool = False,
    ):
        self._constraints[name] = constraint(name, kernel, target, label, weight)

        if log:
            self._metrics.append(name)

    def forward(self, pred, label):
        return torch.sum(
            torch.stack(
                [
                    constraint.weight
                    * constraint.kernel(
                        pred[constraint.target], label[constraint.label]
                    )
                    for constraint in self._constraints.values()
                ]
            )
        )
    
    def get_constraint(self, name):
        return self._constraints[name]


class ExponentialLW:

    def __init__(self, constraint: constraint, gamma: float):
        self._constraint = constraint
        self._gamma = gamma

    def step(self):
        weight = self._constraint.weight
        new_weight = weight * self._gamma + (1 - self._gamma)
        self._constraint.weight = new_weight

