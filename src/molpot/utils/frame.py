import torch
from tensordict import TensorDict
from tensordict.base import _register_tensor_class


class Frame(TensorDict):

    @classmethod
    def from_frames(cls, frames):
        td = cls.maybe_dense_stack(frames)
        return td
    
    def get_frame_list(self):
        return [Frame(frame) for frame in self]
    
_register_tensor_class(Frame)

