import torch


def index_add(
    x: torch.Tensor, dim: int, idx_i: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """
    Sum over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        reduced input

    """
    shape = [dim_size] + list(x.shape[1:])
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return tmp.index_add(dim, idx_i, x)

def index_acc(x: torch.Tensor, dim: int, idx_i: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Accumulate over values with the same indices.

    Args:
        x: input values
        idx_i: index of center atom i
        dim_size: size of the dimension after reduction

    Returns:
        reduced input

    """
    return x.index_add(dim, idx_i, src)