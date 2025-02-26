
from torch import Tensor
from molpot_op.scatter import scatter_sum

def get_natoms_per_batch(atom_batch: Tensor) -> Tensor:
    """Get the number of atoms per batch from the atomic batch mask.

    Args:
        atom_batch (Tensor): atomic batch mask

    Returns:
        Tensor: number of atoms per batch

    Example:
        >>> from torch import tensor
        >>> from molpot.utils import get_natoms_per_batch
        >>> get_natoms_per_batch(tensor([0, 0, 1, 1, 1, 2, 2]))
        tensor([2, 3, 2])
    """
    return atom_batch.bincount()

def batch_add(props: Tensor, atom_batch: Tensor) -> Tensor:
    """Index add properties to the batch.

    Args:
        props (Tensor): properties
        atom_batch (Tensor): atomic batch mask

    Returns:
        Tensor: properties with batch added

    Example:
        >>> from torch import tensor, arange
        >>> from molpot.utils import batch_add
        >>> batch_add(arange(7), tensor([0, 0, 1, 1, 1, 2, 2]))
        tensor([ 1,  9, 11])
    """
    return scatter_sum(
        props, atom_batch, dim=0, dim_size=len(atom_batch.unique())
    )