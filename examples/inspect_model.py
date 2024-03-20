from pathlib import Path

import molpy as mp
import torch
from torchviz import make_dot

import molpot as mpot


def inspect_model():

    model = mpot.Potentials(mpot.potential.classical.pair.LJ126(1, 1), derive_energy=False)
    inputs = {
        mp.Alias.R: torch.rand(10, 3, requires_grad=True),
        mp.Alias.cell: torch.eye(3),
        mp.Alias.pbc: torch.tensor([True, True, True]),
        mp.Alias.atype: torch.zeros(10, dtype=int),
    }
    cutoff = 0.4
    nblist = mpot.neighborlist.TorchNeighborList(cutoff)
    inputs = nblist(inputs)
    output = model(inputs)
    y = output[mp.Alias.energy]
    make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("model", format="png")

if __name__ == "__main__":
    inspect_model()