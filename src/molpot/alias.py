from typing import Union
import torch

AliasKey = tuple[str, str]


class Alias:

    def __init__(
        self,
        name: str,
        key: AliasKey,
        type: type,
        unit: str,
        namespace: str,
        comment: str = "",
        shape: tuple = (),
    ) -> None:
        self.name = name
        self.key = key
        self.type = type
        self.unit = unit
        self.namespace = namespace
        self.shape = shape
        self.comment = comment

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"<Alias: {self.name}>"

    def __eq__(self, o: Union[str, "Alias", tuple]) -> bool:
        if isinstance(o, Alias):
            return self.name == o.name
        else:
            return self.name == o


class NameSpace(dict):

    namespaces = {}

    def __new__(cls, name: str):
        if name not in cls.namespaces:
            ins = super().__new__(cls)
            cls.namespaces[name] = ins
        else:
            ins = cls.namespaces[name]
        return ins

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<NameSpace: {self.name}>"

    def add(self, alias: Alias) -> AliasKey:
        self[alias.name] = alias
        return alias.key
    
    def __getattribute__(self, name: str) -> str:
        if name in NameSpace.namespaces:
            return NameSpace.namespaces[name]
        elif name in self:
            return self[name].key
        return super().__getattribute__(name)

    def set(
        self,
        name: str,
        type_: type,
        unit: str,
        comment: str,
        shape: tuple = (),
    ) -> AliasKey:
        namespace = self.name
        key = (namespace, name)
        return self.add(
            Alias(
                name=name,
                key=key,
                type=type_,
                unit=unit,
                namespace=namespace,
                comment=comment,
                shape=shape,
            )
        )


# atoms section
atoms_ns = NameSpace("atoms")
atomid = atoms_ns.set("idx", int, "unit", "atom index")
molid = atoms_ns.set("molid", int, "unit", "molecule index")
xyz = R = atoms_ns.set("xyz", torch.Tensor, "unit", "atom coordinates")
Z = atoms_ns.set("Z", int, "unit", "atomic number")
atom_batch_mask = atoms_ns.set("atomic_batch_mask", torch.Tensor, "unit", "atoms batch mask")

# cell section
cell_ns = NameSpace("cell")
pbc = cell_ns.set("pbc", torch.Tensor, "unit", "periodic boundary condition")
cell = cell_ns.set("matrix", torch.Tensor, "unit", "cell matrix")

# bonds section
bonds_ns = NameSpace("bonds")
bond_i = bonds_ns.set("i", int, "unit", "bond atom index i")
bond_j = bonds_ns.set("j", int, "unit", "bond atom index j")
bond_diff = bonds_ns.set("diff", torch.Tensor, "angstrom", "bond displacement")
bond_dist = bonds_ns.set("dist", torch.Tensor, "angstrom", "bond distance")

# pairs section
pairs_ns = NameSpace("pairs")
pair_i = pairs_ns.set("i", int, "unit", "pair atom index i")
pair_j = pairs_ns.set("j", int, "unit", "pair atom index j")
pair_diff = pairs_ns.set("diff", torch.Tensor, "angstrom", "pair displacement")
pair_dist = pairs_ns.set("dist", torch.Tensor, "angstrom", "pair distance")
pair_offset = pairs_ns.set("offset", torch.Tensor, "unit", "offsets")

# prop section
props_ns = NameSpace("props")
n_atoms = props_ns.set("n_atoms", int, "unit", "number of atoms")
n_pairs = props_ns.set("n_pairs", int, "unit", "number of pairs")