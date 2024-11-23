from molpot.utils.locality import get_neighbor_pairs
from molpot import alias, Config
import torch


class Transform(torch.nn.Module):
    pass

class NeighborList(Transform):
    def __init__(self, cutoff, max_num_pairs=-1, check_errors=False, padding=True):
        self.cutoff = cutoff
        self.max_num_pairs = max_num_pairs
        self.check_errors = check_errors
        self.padding = padding

    def __call__(self, tensordict):

        cell = tensordict[alias.cell]
        xyz = tensordict[alias.xyz]

        if torch.allclose(cell, torch.zeros_like(cell)):
            cell = None

        neighbors, deltas, distances, number_found_pairs = get_neighbor_pairs(
            xyz, self.cutoff, box_vectors=cell
        )
        if self.padding:
            mask = neighbors[0] > -1
            pair_i = neighbors[0][mask].to(torch.int64)
            pair_j = neighbors[1][mask].to(torch.int64)
            deltas = deltas[mask].to(Config.ftype)
            distances = distances[mask].to(Config.ftype)
        else:
            pair_i = neighbors[0].to(torch.int64)
            pair_j = neighbors[1].to(torch.int64)
            deltas = deltas.to(Config.ftype)
            distances = distances.to(Config.ftype)
            
        tensordict[alias.pair_i] = pair_i  # for scatter 
        tensordict[alias.pair_j] = pair_j  # for scatter 
        tensordict[alias.pair_diff] = deltas
        tensordict[alias.pair_dist] = distances
        
        return tensordict