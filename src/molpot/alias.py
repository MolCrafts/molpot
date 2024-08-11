from typing import Union
import torch

class Alias:

    def __init__(self, name: str, key: str, type: type, unit: str, comment: str, shape: tuple=()) -> None:
        self.name = name
        self.key = key
        self.type = type
        self.unit = unit
        self.shape = shape
        self.comment = comment

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"<Alias: {self.name}>"
    
    def __eq__(self, o: Union[str, "Alias", tuple]) -> bool:
        if isinstance(o, Alias):
            return self.name == o.name
        elif isinstance(o, (tuple, str)):
            return self.name == o
        else:
            return False

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
        self[alias.name] = alias
        return alias

    def set(self, name:str, key: str, type_: type, unit: str, comment: str, shape: tuple=(), namespace:str="default"):
        return self.add(Alias(name, key, type_, unit, comment, shape))

# atoms section
default_ns = NameSpace("default")
n_atoms = default_ns.set("n_atoms", ("atoms", "n_atoms"), int, "unit", "number of atoms")
atomid = default_ns.set("idx", ("atoms", "idx"), int, "unit", "atom index")
molid = default_ns.set("molid", ("atoms", "molid"), int, "unit", "molecule index")
xyz = R = default_ns.set("xyz", ("atoms", "xyz"), float, "unit", "atom coordinates")
Z = default_ns.set("Z", ("atoms", "Z"), int, "unit", "atomic number")

# box section
pbc = default_ns.set("pbc", ("box", "pbc"), torch.Tensor, "unit", "periodic boundary condition")

box_matrix = default_ns.set("box_matrix", ("box", "matrix"), torch.Tensor, "unit", "cell matrix")

# bonds section
bond_i = default_ns.set("bond_i", ("bonds", "i"), int, "unit", "bond atom index i")
bond_j = default_ns.set("bond_j", ("bonds", "j"), int, "unit", "bond atom index j")
bond_diff = default_ns.set("diff", ("bond", "diff"), torch.Tensor, "angstrom", "bond displacement")
bond_dist = default_ns.set("dist", ("bond", "dist"), torch.Tensor, "angstrom", "bond distance")

# pairs section
pair_i = default_ns.set("pair_i", ("pairs", "i"), int, "unit", "pair atom index i")
pair_j = default_ns.set("pair_j", ("pairs", "j"), int, "unit", "pair atom index j")
pair_diff = default_ns.set("pair_diff", ("pairs", "diff"), torch.Tensor, "angstrom", "pair displacement")
pair_dist = default_ns.set("pair_dist", ("pairs", "dist"), torch.Tensor, "angstrom", "pair distance")
pair_offset = default_ns.set("pair_offset", ("pairs", "offset"), torch.Tensor, "unit", "offsets")

# global section
