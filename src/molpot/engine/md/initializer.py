from .handler import MDHandler, MDMainEvents, Events

import torch

class Initializer(MDHandler):

    def __init__(self):
        super().__init__(
            {Events.STARTED},
            (0,)
        )

class Zeros(Initializer):

    def __init__(self, *targets: str):
        self.targets = targets
        super().__init__()

    def on_started(self, engine):
        for target in self.targets:
            engine.state.frame[target] = 0
        return engine