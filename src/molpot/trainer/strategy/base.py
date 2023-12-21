# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2023-12-07
# version: 0.0.1

class Strategy:

    def __call__(self) -> bool:
        raise NotImplementedError
    
    @property
    def name(self):
        return self.__class__.__name__
    
class StrategyManager:
    
    def __init__(self):
        self.strategies = []
    
    def add(self, strategy:Strategy):
        self.strategies.append(strategy)
    
    def __call__(self, step:int) -> bool:
        for strategy in self.strategies:
            if strategy(step):
                return True
        return False