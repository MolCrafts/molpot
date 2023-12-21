from functools import partial
from torchdata.datapipes.iter import IterDataPipe
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

__all__ = ['DataLoader', 'create_dataloader']

class DataLoader(DataLoader2):
    pass

def create_dataloader(datapipe: IterDataPipe, nworkers:int=0) -> DataLoader:

    datapipe = datapipe.collate_data()

    if nworkers:
        rs = MultiProcessingReadingService(nworkers)
        return DataLoader(datapipe, reading_service=rs)
    else:
        return DataLoader(datapipe)
