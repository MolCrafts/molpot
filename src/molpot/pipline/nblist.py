from molpot.utils.locality import get_neighbor_pairs
from molpot import alias, Config
import torch


class Transform(torch.nn.Module):
    pass

class NeighborList(Transform):
    def __init__(self, cutoff, max_num_pairs=-1, check_errors=False):
        self.cutoff = cutoff
        self.max_num_pairs = max_num_pairs
        self.check_errors = check_errors

    def __call__(self, tensordict):

        cell = tensordict[alias.cell]
        xyz = tensordict[alias.xyz]

        neighbors, deltas, distances, number_found_pairs = get_neighbor_pairs(
            xyz, self.cutoff, box_vectors=cell
        )

        tensordict[alias.pair_i] = neighbors[0].to(torch.int64)  # for scatter 
        tensordict[alias.pair_j] = neighbors[1].to(torch.int64)  # for scatter 
        tensordict[alias.pair_diff] = deltas.to(Config.ftype)
        tensordict[alias.pair_dist] = distances.to(Config.ftype)
        return tensordict