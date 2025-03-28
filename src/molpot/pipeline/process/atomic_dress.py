import torch

from .base import Process, ProcessType


class AtomicDress(Process):

    type = ProcessType.ALL

    def __init__(self, dress_key: str):
        """
        Fit the atomic energy with an element-dependent atomic dress.
        
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
        self.key = dress_key

    def forward(self, frames):

        Z = torch.cat([frame["atoms", "Z"] for frame in frames], dim=0)
        elems = torch.unique(Z)
       
        x = []
        y = []
        z = []
        for frame in frames:
            # count = torch.bincount(frame["atoms", "Z"]-1, minlength=max(elems))
            count = frame["atoms", "Z"][:, None] == elems[None, ...]
            x.append(count.sum(dim=0).float())
            y.append(frame[self.key])
            z.append(frame["atoms", "Z"])

        x = torch.stack(x)
        x = torch.concat([x, torch.zeros((x.shape[0], 1))], dim=1)
        y = torch.stack(y)
        beta = torch.linalg.pinv(x.T @ x) @ x.T @ y
        self.atomic_dress = {e.item(): beta[i].item() for i, e in enumerate(elems)}
        
        # Compute the residual error (difference between the prediction and true values).
        error = torch.matmul(x, beta) - y

        # substract the fitted energy from the true energy
        for frame in frames:
            frame[self.key] -= torch.sum(torch.tensor([self.atomic_dress[z.item()] for z in frame["atoms", "Z"]]))
        return frames
