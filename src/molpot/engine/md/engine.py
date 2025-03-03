from torch.nn import Module
from .handler import MDHandler, Potential
from ..base import MolpotEngine
from ignite.engine import Engine, Events
from pathlib import Path
from .main import main_process, MDMainEvents
import torch
from itertools import islice, cycle

class MoleculeDymanics(MolpotEngine):

    def __init__(self, grad_required=False, work_dir=Path.cwd()):

        super().__init__(work_dir=work_dir)

        self.grad_required = grad_required

        main_engine = Engine(
                main_process(
                    gradients_required=self.grad_required
                )
            )
        
        main_engine.register_events(*MDMainEvents)

        self.add_engine(
            "main",
            main_engine
        )

        # main_engine.add_event_handler(Events.STARTED, initialize system)

    def add_module(self, module: MDHandler):
        
        module.attach(self.main)

    def add_potential(self, potential: Module):

        self.add_module(Potential(potential))

    def run(self, frame, max_steps):

        self.main.add_event_handler(
            Events.ITERATION_STARTED,
            lambda engine: (
                engine.terminate() if engine.state.iteration >= max_steps else None
            ),
        )


        self.main.run(islice(cycle(frame), max_steps), max_epochs=torch.inf)