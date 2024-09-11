import torch

from tensordict.tensordict import TensorDict, LazyStackedTensorDict

from typing import Sequence
from molpot.alias import Alias

class Frames(LazyStackedTensorDict):
    pass

class Frame(TensorDict):

    def __init__(self, 
        source: dict[str] = None,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: None = None,
        names: Sequence[str] | None = None,
        non_blocking: bool = None,
        lock: bool = False,
        **kwargs,
    ):
        default = {
            'atoms': {},
            'bonds': {},
            'angles': {},
            'dihedrals': {},
            'impropers': {},
            'pairs': {},
            'box': {},
            'props': {},  # global features
            'targets': {},
            'predicts': {}
        }
        if source is not None:
            default.update(source)
        super().__init__(default, batch_size, device, names, non_blocking, lock, **kwargs)

    @classmethod
    def maybe_dense_stack(cls, input, dim=0, *, out=None, **kwargs):
        return Frames.maybe_dense_stack(
            input, dim=dim, out=out, **kwargs
        )
