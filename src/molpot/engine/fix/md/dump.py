import chemfiles as chfl
import numpy as np
import torch

from ..base import Fix


class Dump:

    def __init__(self, every_n_steps: int):
        self.every_n_steps = every_n_steps
        self.priority = 10

    def finalize(self, engine, status, inputs, outputs):
        pass

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
    
class DumpNPZ(Dump):

    def __init__(self, name: str, every_n_steps: int):
        super().__init__(every_n_steps)
        self._traj = []
        self._name = name

    def __call__(self, engine, status, inputs, outputs):
            
        if status['current_step'] % self.every_n_steps == 0:
            self._traj.append(outputs['atoms']['xyz'].detach().cpu().numpy())
        return inputs, outputs
    
    def finalize(self, engine, status, inputs, outputs):
        np.savez(self._name, traj=self._traj)
        return inputs, outputs
    
class DumpTensor(Dump):

    def __init__(self, out: torch.Tensor, every_n_steps: int):
        super().__init__(every_n_steps)
        self._out = out
        self._tmp = []

    def __call__(self, engine, status, inputs, outputs):
                
        if status['current_step'] % self.every_n_steps == 0:
            self._tmp.append(outputs[self._out].detach().cpu().numpy())
        return inputs, outputs
    
    def finalize(self, engine, status, inputs, outputs):

        self._out[:] = torch.tensor(self._tmp)

        return inputs, outputs


class DumpXTC(Dump):

    def __init__(self, name: str, every_n_steps: int):
        super().__init__(every_n_steps)
        self._name = name
        self._file = chfl.Trajectory(name, mode='w', format='XTC')

    def __call__(self, engine, status, inputs, outputs):
            
        if status['current_step'] % self.every_n_steps == 0:
            frame = chfl.Frame()
            frame.resize(outputs['n_atoms'])
            frame.positions[:] = outputs['atoms']['xyz'].detach().cpu().numpy()
                
            self._file.write(frame)
        return inputs, outputs
    
    def finalize(self, engine, status, inputs, outputs):
        self._file.close()
        return super().finalize(engine, status, inputs, outputs)