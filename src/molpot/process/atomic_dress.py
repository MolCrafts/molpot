import torch
import torch.nn as nn
from molpot import alias

import torch
from torch.utils.data import DataLoader


class AtomicDress(nn.Module):

    def __init__(self):
        """
        Fit the atomic energy with an element–dependent atomic dress.
        
        Args:
            dataset: A PyTorch dataset or iterable. Each item is expected to be a dict with at least:
                - 'elems': a tensor of element numbers for each atom,
                - 'ind_1': a tensor where the first column gives the segment (e.g. structure) index,
                - key (default 'e_data'): the property to fit (e.g. the total energy of the structure).
            elems: A list of element numbers (or identifiers) that will be used to form the linear model.
            key: The dictionary key for the property to be fitted.
        
        Returns:
            atomic_dress: A dictionary mapping each element (from elems) to its fitted coefficient.
            error: A torch tensor with the residual error (difference between model prediction and true values).
        """
        super().__init__()

    def forward(self, frames):

        Z = torch.cat([frame["atoms", "Z"] for frame in frames], dim=0)
        elems = torch.unique(Z)
       
        x = []
        y = []
        for frame in frames:
            count = torch.bincount(frame["atoms", "Z"]-1, minlength=len(elems))
            x.append(count.float())
            y.append(frame[("labels", "energy")])
        
        x = torch.stack(x)
        y = torch.stack(y)
        beta = torch.linalg.pinv(x.T @ x) @ x.T @ y
        # Build a dictionary mapping each element to its fitted coefficient.
        self.atomic_dress = {elem+1: e.item() for elem, e in enumerate(beta)}
        
        # Compute the residual error (difference between the prediction and true values).
        error = torch.matmul(x, beta) - y

        # substract the fitted energy from the true energy
        for frame in frames:
            frame[("labels", "energy")] -= torch.sum(beta[frame["atoms", "Z"]-1])
        print(torch.sqrt(torch.mean(error**2)))
        return frames