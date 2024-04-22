# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-12
# version: 0.0.1

from torch import nn

def multi_targets(loss_fn, weights, targets):
    def multi_traget_loss_fn(input, target):
        loss = 0
        for weight, (output, label) in zip(weights, targets):
            loss += weight * loss_fn(input[output], target[label])
        return loss
    return multi_traget_loss_fn