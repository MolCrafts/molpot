import torch
from tensordict import TensorDict
from tensordict.base import _register_tensor_class
from molpot import alias

class Frame(TensorDict):

    @classmethod
    def from_frames(cls, frames):
        td = cls.maybe_dense_stack(frames)
        return td
    
    def get_frame_list(self):
        return [Frame(frame) for frame in self]
    
    @property
    def n_atoms(self):
        return self[alias.R].shape[0]
    
    @property
    def n_molecules(self):
        return self[alias.molid].shape[0]
    
_register_tensor_class(Frame)

