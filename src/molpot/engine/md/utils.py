import molpot as mpot
import torch

def create_lj_system(n_atoms: int | list[int], masses: float | list[float], box):
    
    def create_one_type(n_atoms, mass, typeid):
        frame = mpot.Frame()
        frame[mpot.alias.R] = torch.rand(n_atoms, 3)
        frame[mpot.alias.molid] = torch.zeros(n_atoms, dtype=torch.int) + 1
        frame[mpot.alias.atom_type_id] = torch.zeros(n_atoms, dtype=torch.int) + typeid
        frame[mpot.alias.atom_mass] = torch.zeros(n_atoms, dtype=torch.float) + mass
        return frame
    
    if isinstance(n_atoms, int):
        return create_one_type(n_atoms, masses, 0)
    else:
        frames = []
        for i, n in enumerate(n_atoms):
            frames.append(create_one_type(n, masses[i], i))
        return mpot.Frame.from_frames(frames)