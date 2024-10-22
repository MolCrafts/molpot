from torch import nn

def loss(kernel: nn.Module, input_key: str, target_key: str):
    def _loss(input_, target):
        inputs = input_[input_key]
        targets = target[target_key]
        return kernel(inputs, targets)

    return _loss