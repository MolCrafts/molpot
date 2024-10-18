import torch

AliasKey = tuple[str, str] | str


class Alias:

    def __init__(
        self,
        name: str,
        comment: str,
        dtype: type,
        unit: str|None=None,
        shape: tuple = (),
        namespace: str = "",
        category: str = "",
    ) -> None:
        self.name = name
        self.unit = unit
        self.comment = comment

        self.dtype = dtype
        self.shape = shape

        self.namespace = namespace
        self.category = category

    @property
    def key(self) -> AliasKey:
        return tuple([part for part in (self.namespace, self.category, self.name) if part])
    
    @property
    def is_array(self) -> bool:
        return len(self.shape) > 0


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
        alias.namespace = self.name
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
        comment: str,
        dtype: type,
        unit: str|None=None,
        shape: tuple = (),
        category: str = "",
    ) -> AliasKey:
        return self.add(
            Alias(
                name=name,
                comment=comment,
                unit=unit,
                dtype=dtype,
                shape=shape,
                namespace=self.name,
                category=category,
            )
        )


# atoms section
atoms_ns = NameSpace("atoms")
atomid = atoms_ns.set("idx", "atom index", int)
molid = atoms_ns.set("molid", "molecule index", int)
xyz = R = atoms_ns.set("xyz", "atom coordinates", float, shape=(None, 3))
Z = atoms_ns.set("Z", "atomic number", int)
atom_batch_mask = atoms_ns.set("atomic_batch_mask", "atoms batch mask", int)

# # cell section
# cell_ns = NameSpace("cell")
# pbc = cell_ns.set("pbc", torch.Tensor, None, "periodic boundary condition")
# cell = cell_ns.set("matrix", torch.Tensor, None, "cell matrix")

# # bonds section
bonds_ns = NameSpace("bonds")
bond_i = bonds_ns.set("i", "bond atom index i", int)
bond_j = bonds_ns.set("j", "bond atom index j", int)
bond_diff = bonds_ns.set("diff", "bond displacement", float, unit="angstrom")
bond_dist = bonds_ns.set("dist", "bond distance", float, unit="angstrom")

# # pairs section
pairs_ns = NameSpace("pairs")
pair_i = pairs_ns.set("i", "pair atom index i", int)
pair_j = pairs_ns.set("j", "pair atom index j", int)
pair_diff = pairs_ns.set("diff", "pair displacement", float, shape=(None, 3))
pair_dist = pairs_ns.set("dist", "pair distance", float, shape=(None, 3))
# pair_offset = pairs_ns.set("offset", "offsets", int)

# prop section
props_ns = NameSpace("props")
n_atoms = props_ns.set("n_atoms", "number of atoms", int)
n_pairs = props_ns.set("n_pairs", "number of pairs", int)