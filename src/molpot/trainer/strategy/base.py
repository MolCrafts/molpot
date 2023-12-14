# author: Roy Kid
# contact: lijichen365@gmail.com
# date: 2023-12-07
# version: 0.0.1

class Strategy:
    
    def __init__(self, name:str):
        self.name = name

    def __call__(self) -> bool:
        raise NotImplementedError
    
class PlannedStop(Strategy):

    def __init__(self, nstep:int):
        super().__init__("PlannedStop")
        self.nstep = nstep

    def __call__(self, step:int) -> bool:
        if step >= self.nstep:
            return True
        return False