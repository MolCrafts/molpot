# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-07
# version: 0.0.1

from torchdata.datapipes.iter import IterDataPipe

class Engine:

    def prepare(self, *args, **kwargs) -> IterDataPipe:
        raise NotImplementedError

    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError