import torch
from torch import nn

# def loss(kernel: nn.Module, keys: tuple[str, str, float]) -> callable:
#     def _loss(input_, target):
#         return reduce(add, [weight * kernel(input_[in_key], target[out_key]) for in_key, out_key, weight in keys])
#     return _loss


class MultiTargetLoss(nn.Module):
    def __init__(self, kernel: nn.Module, keys: tuple[str, str, float]):
        super().__init__()
        self.kernel = kernel
        self.keys = keys

    def forward(self, input_, target):
        tmp = torch.sum(
            torch.stack(
                [
                    weight * self.kernel(input_[in_key], target[out_key])
                    for in_key, out_key, weight in self.keys
                ],
                dim=-1,
            ),
            dim=-1,
        )
        print(tmp)
        return tmp