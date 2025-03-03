from ..base import MolpotEngine
from ignite.engine import Engine, Events
from pathlib import Path
from .main import main_process, MDMainEvents

class MoleculeDymanics(MolpotEngine):

    def __init__(self, integrator, calculator, neighborlist, grad_required=False, work_dir=Path.cwd()):

        super().__init__(work_dir=work_dir)

        self.integrator = integrator
        self.calculator = calculator
        self.neighborlist = neighborlist
        self.grad_required = grad_required

        main_engine =             Engine(
                main_process(
                    integrator=self.integrator,
                    calculator=self.calculator,
                    neighborlist=self.neighborlist,
                    gradients_required=self.grad_required
                )
            )
        
        main_engine.register_events(*MDMainEvents)

        self.add_engine(
            "main",
            main_engine
        )

        # main_engine.add_event_handler(Events.STARTED, initialize system)