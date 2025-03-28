from enum import Enum
from torch import nn

class ProcessType(Enum):

    ONE = "one"
    ALL = "all"

class Process(nn.Module): ...

class ProcessManager:

    def __init__(self):

        self._process_one = nn.Sequential()
        self._process_all = nn.Sequential()

    def append(self, process: Process):

        if process.type == ProcessType.ONE:
            self._process_one.append(process)

        elif process.type == ProcessType.ALL:
            self._process_all.append(process)
        else:
            raise ValueError(f"Unknown process type: {process.type}")
        
    def process_one(self, frame):
        """Process a single frame."""
        return self._process_one(frame)
    
    def process_all(self, frames):
        """Process all frames."""
        return self._process_all(frames)