import torch

from tensordict import LazyStackedTensorDict, TensorDict
from tensordict._td import T, CompatibleType, DeviceType

from typing import Sequence
from molpot.alias import Alias

class Frames(LazyStackedTensorDict):
    pass

class Frame(TensorDict):

    def __init__(self, 
        source: T | dict[str, CompatibleType] = None,
        batch_size: Sequence[int] | torch.Size | int | None = None,
        device: DeviceType | None = None,
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
    
    def __getitem__(self, key: str) -> T:
        if isinstance(key, Alias):
            key = key.key
        return super().__getitem__(key)
    
    def __setitem__(self, key: str, value: T) -> None:
        if isinstance(key, Alias):
            key = key.key
        super().__setitem__(key, value)

    def __contains__(self, key: str) -> bool:
        if isinstance(key, Alias):
            key = key.key
        return super().__contains__(key)