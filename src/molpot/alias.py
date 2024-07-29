from typing import Union
import torch

class Alias:

    def __init__(self, name: str, type: type, unit: str, comment: str, shape: tuple=(), namespace: str = "default") -> None:
        self.name = name
        self.type = type
        self.unit = unit
        self.shape = shape
        self.comment = comment
        self.namespace = namespace

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"<Alias: {self.name}>"
    
    def __eq__(self, o: Union[str, "Alias"]) -> bool:
        if isinstance(o, Alias):
            return self.name == o.name
        else:
            return self.name == o


class NameSpace(dict):

    namespaces = {}

    def __new__(cls, name:str):
        if name not in cls.namespaces:
            ins = super().__new__(cls)
            cls.namespaces[name] = ins
        else:
            ins = cls.namespaces[name]
        return ins
    
    def __init__(self, name:str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<NameSpace: {self.name}>"
    
    def add(self, alias:Alias):
        alias.namespace = self.name
        self[alias.name] = alias

    def set(self, alias: str, type_: type, unit: str, comment: str, shape: tuple=()):
        self.add(Alias(alias, type_, unit, comment, self.name, shape))

n_atoms = Alias("n_atoms", int, "unit", "number of atoms")
atomid = Alias("idx", int, "unit", "atom index")
xyz = R = Alias("xyz", float, "unit", "atom coordinates")
Z = Alias("Z", int, "unit", "atomic number")
pbc = Alias("pbc", torch.Tensor, "unit", "periodic boundary condition")
cell = Alias("cell", torch.Tensor, "unit", "cell matrix")
molid = Alias("molid", int, "unit", "molecule index")
offsets = Alias("offsets", torch.Tensor, "unit", "offsets")
pair_i = Alias("pair_i", int, "unit", "pair atom index i")
pair_j = Alias("pair_j", int, "unit", "pair atom index j")
pair_diff = Alias("pair_diff", torch.Tensor, "angstrom", "pair displacement")
pair_dist = Alias("pair_dist", torch.Tensor, "angstrom", "pair distance")
bond_i = Alias("bond_i", int, "unit", "bond atom index i")
bond_j = Alias("bond_j", int, "unit", "bond atom index j")
bond_diff = Alias("bond_diff", torch.Tensor, "angstrom", "bond displacement")
bond_dist = Alias("bond_dist", torch.Tensor, "angstrom", "bond distance")
