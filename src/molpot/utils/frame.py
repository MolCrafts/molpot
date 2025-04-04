from molpot_op.scatter import scatter_sum
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
        if alias.molid in self:
            return self[alias.molid].unique().shape[0]
        else:
            return torch.arange(self.n_atoms)
    
    def sum_atoms(self, v):
        """
        Auxiliary routine for summing atomic contributions for each molecule.

        Args:
            x (torch.Tensor): Input tensor of the shape ( : x (n_molecules * n_atoms) x ...)

        Returns:
            torch.Tensor: Aggregated tensor of the shape ( : x n_molecules x ...)
        """
        molid = self[alias.molid] if alias.molid in self else torch.arange(self.n_atoms)
        return scatter_sum(
            v, molid, dim=0, dim_size=len(molid.unique())
        )
    
    def expand_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for expanding molecular contributions over the corresponding atoms.

        Args:
            x (torch.Tensor): Tensor of the shape ( : x n_molecules x ...)

        Returns:
            torch.Tensor: Tensor of the shape ( : x (n_molecules * n_atoms) x ...)
        """
        molid = self[alias.molid] if alias.molid in self else torch.arange(self.n_atoms)
        return x[:, molid, ...]
    
    def mean_atoms(self, x: torch.Tensor):
        """
        Auxiliary routine for computing mean over atomic contributions for each molecule.

        Args:
            x (torch.Tensor): Input tensor of the shape ( : x (n_molecules * n_atoms) x ...)

        Returns:
            torch.Tensor: Aggregated tensor of the shape ( : x n_molecules x ...)
        """
        return self.sum_atoms(x) / self.n_atoms[None, :, None]
    
    @property
    def center_of_mass(self):
        com = self.sum_atoms(
            alias.R
        ) / self.sum_atoms(
            alias.atom_mass
        )
        return com
    
    def remove_center_of_mass(self):
        com = self.center_of_mass
        self[alias.R] -= self.expand_atoms(com)
        return self
    
    def remove_translation(self):
        self[alias.atom_momentum] -= self.expand_atoms(
            self[alias.atom_momentum].sum(dim=0) / self.n_molecules
        )
        return self
    
    # def remove_com_rotation(self):

    
_register_tensor_class(Frame)

