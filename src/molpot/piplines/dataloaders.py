from functools import partial

from torchdata.dataloader2 import (DataLoader2, InProcessReadingService,
                                   MultiProcessingReadingService)
from torchdata.datapipes.iter import IterDataPipe


class DataLoader(DataLoader2):
    pass


def create_dataloader(datapipe: IterDataPipe, nworkers: int = 0) -> DataLoader:

    datapipe = datapipe.collate_data()

    if nworkers:
        rs = MultiProcessingReadingService(nworkers)
        return DataLoader(datapipe, reading_service=rs)
    else:
        return DataLoader(datapipe, reading_service=InProcessReadingService())
