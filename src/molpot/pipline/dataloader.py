from torch.utils.data import DataLoader
from typing import Sequence
from molpot import Frame

def collate_frame(batch: Sequence[Frame]):
    return Frame.maybe_dense_stack(batch, dim=0)
    
class DataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(collate_fn=collate_frame, *args, **kwargs)
        