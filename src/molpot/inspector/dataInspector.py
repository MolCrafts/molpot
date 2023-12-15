# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-15
# version: 0.0.1

class DataInspector:

    def __init__(self, dataloader):

        self.dataloader = dataloader

    def inspect(self):

        for batch in self.dataloader:
            