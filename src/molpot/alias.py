from typing import Union
import torch

class Alias:

    name: str
    type_: type
    unit: str
    comment: str
    namespace: str = "default"

    @staticmethod
    def name2key(name: str) -> str:
        return f"_{name}"

    def __init__(self, name: str, type_: type, unit: str, comment: str, namespace: str = "default") -> None:
        self.name = name
        self.key = Alias.name2key(name)
        self.type = type_
        self.unit = unit
        self.comment = comment
        self.namespace = namespace

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"<Alias: {self.name}>"
    
    def __eq__(self, o: Union[str, "Alias"]) -> bool:
        return self.name == o or self.name == o.name


class NameSpace(dict):

    namespaces = {
        'default': {}
    }

    def __new__(cls, name:str):
        if name not in cls.namespaces:
            cls.namespaces[name] = {}
            return super().__new__(cls)
        else:
            return cls.namespaces[name]

    def __init__(self, name:str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<NameSpace: {self.name}>"
    
    def add(self, alias:Alias):
        alias.namespace = self.name
        self[alias.name] = alias

    def set(self, alias: str, type_: type, unit: str, comment: str):
        self.add(Alias(alias, type_, unit, comment))

n_atoms = Alias("n_atoms", int, "unit", "number of atoms")
atomid = Alias("idx", int, "unit", "atom index")
xyz = Alias("xyz", float, "unit", "atom coordinates")
Z = Alias("Z", int, "unit", "atomic number")
pbc = Alias("pbc", torch.Tensor, "unit", "periodic boundary condition")
cell = Alias("cell", torch.Tensor, "unit", "cell matrix")
molid = Alias("molid", int, "unit", "molecule index")
pair_i = Alias("pair_i", int, "unit", "atom index i")
pair_j = Alias("pair_j", int, "unit", "atom index j")