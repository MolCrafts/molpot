import torch
import torch.nn as nn
import molpot as mp

class Potential(nn.Module):

    def forward(self, inputs:dict, forces:bool=False, stress:bool=False) -> dict:
        pass
    
class PotentialDict(nn.ModuleDict):

    def forward(self, inputs:dict, forces:bool=False, stress:bool=False) -> dict:
        pass
    
class PotentialSeq(nn.Sequential):

    def forward(self, inputs:dict, forces:bool=False, stress:bool=False) -> dict:
        pass