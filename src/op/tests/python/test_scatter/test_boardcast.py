from itertools import product
import molpot_op
import pytest
import torch
from .utils import devices


reductions = ["sum", "add"]

@pytest.mark.parametrize('reduce,device', product(reductions, devices))
def test_broadcasting(reduce, device):
    B, C, H, W = (4, 3, 8, 8)
    fn = getattr(molpot_op.scatter, 'scatter_' + reduce)
    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = fn(src, index, dim=2, dim_size=H)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (B, 1, H, W)).to(device, torch.long)
    out = fn(src, index, dim=2, dim_size=H)
    assert out.size() == (B, C, H, W)

    src = torch.randn((B, C, H, W), device=device)
    index = torch.randint(0, H, (H, )).to(device, torch.long)
    out = fn(src, index, dim=2, dim_size=H)
    assert out.size() == (B, C, H, W)