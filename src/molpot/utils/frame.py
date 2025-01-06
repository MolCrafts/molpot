import torch
from tensordict.tensordict import TensorDict, LazyStackedTensorDict

class Frame(TensorDict):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_frames(cls, frames, layout=torch.jagged):
        return cls.maybe_dense_stack(frames).densify(layout=layout)