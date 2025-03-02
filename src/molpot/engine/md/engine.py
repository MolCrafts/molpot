from ..base import MolpotEngine
from ignite.engine import Engine, Events
from .main import main_process, MDMainEvents

class MolPotMD(MolpotEngine):

    def __init__(self, integrator, calculator, neighborlist, grad_required=False):

        super().__init__()

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