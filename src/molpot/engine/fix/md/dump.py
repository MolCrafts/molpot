import numpy as np
from ..base import Fix

class Dump:

    def __init__(self, every_n_steps: int):
        self.every_n_steps = every_n_steps
        self.priority = 10
        # TODO: manage latest queue

class DumpXYZ(Dump):

    def __init__(self, every_n_steps: int):
        super().__init__(every_n_steps)

    def __call__(self, engine, status, inputs, outputs):
            
        if status['current_step'] % self.every_n_steps == 0:
            xyz = outputs['atoms']['xyz'].detach().cpu().numpy()
            with open(f"dump_{status['current_step']}.xyz", 'w') as f:
                f.write(f"{len(xyz)}\n")
                f.write(f"Step {status['current_step']}\n")
                for i, atom in enumerate(xyz):
                    f.write(f"X {atom[0]} {atom[1]} {atom[2]}\n")
        return inputs, outputs