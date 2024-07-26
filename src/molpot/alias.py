from typing import Union
import torch

class Alias:

    @staticmethod
    def name2key(name: str) -> str:
        return f"_{name}"

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
long_pair_i = Alias("long_pair_i", int, "unit", "long-range pair atom index i")
long_pair_j = Alias("long_pair_j", int, "unit", "long-range pair atom index j")
R = Alias("R", torch.Tensor, "angstrom", "atomic coordinates")
offsets = Alias("offsets", torch.Tensor, "unit", "offsets")
pair_i = Alias("pair_i", int, "unit", "pair atom index i")
pair_j = Alias("pair_j", int, "unit", "pair atom index j")
pair_i_lr = Alias("pair_i_lr", int, "unit", "long-range pair atom index i")
pair_j_lr = Alias("pair_j_lr", int, "unit", "long-range pair atom index j")
d_ij = Alias("d_ij", torch.Tensor, "angstrom", "pair displacement")
dl_ij = Alias("dl_ij", torch.Tensor, "angstrom", "long-range pair displacement")
r_ij = Alias("r_ij", torch.Tensor, "angstrom", "pair distance")
rl_ij = Alias("rl_ij", torch.Tensor, "angstrom", "long-range pair distance")