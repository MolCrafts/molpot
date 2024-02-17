"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from molpy import alias
import numpy as np

__all__ = ["alias"]
# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple
from typing import Any
import numpy as np
from dataclasses import dataclass

class Alias:

    @dataclass
    class Item:
        alias: str
        key: str
        type: Any
        unit: str
        comment: str

        def __hash__(self) -> int:
            return hash(self.key)

        def __repr__(self) -> str:
            return f"<{self.alias}>"

    _scopes: dict[str, dict] = {'default': {
            "timestep": Item("timestep", "_ts", int, "fs", "time step"),
            "name": Item("name", "_name", str, None, "atomic name"),
            "natoms": Item("natoms", "_natoms", int, None, "number of atoms"),
            "xyz": Item("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "R": Item("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "cell": Item("cell", "_cell", np.ndarray, "angstrom", "unit cell"),
            "energy": Item("energy", "_energy", float, "meV", "energy"),
            "forces": Item("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
            "charge": Item("charge", "_charge", float, "C", "charge"),
            "mass": Item("mass", "_mass", float, None, ""),
            "stress": Item("stress", "_stress", np.ndarray, "GPa", "stress"),
            "idx": Item("idx", "_idx", int, None, ""),
            "Z": Item("Z", "_atomic_numbers", int, None, "nuclear charge"),
            "atype": Item("atype", "_atomic_types", int, None, "atomic type"),
            "idx_m": Item("idx_m", "_idx_m", int, None, "indices of systems"),
            "idx_i": Item("idx_i", "_idx_i", int, None, "indices of center atoms"),
            "idx_j": Item("idx_j", "_idx_j", int, None, "indices of neighboring atoms"),
            "idx_i_lr": Item("idx_i_lr", "_idx_i_lr", int, None, "indices of center atoms for # long-range"),
            "idx_j_lr": Item("idx_j_lr", "_idx_j_lr", int, None, "indices of neighboring atoms for # long-range"),
            "offsets": Item("offsets", "_offsets", int, None, "cell offset vectors"),
            "Rij": Item("Rij", "_Rij", np.ndarray, "angstrom", "vectors pointing from center atoms to neighboring atoms"),
            "dist": Item("dist", "_dist", np.ndarray, "angstrom", "distances between center atoms and neighboring atoms"),
            "pbc": Item("pbc", "_pbc", np.ndarray, None, "periodic boundary conditions"),
            "T0": Item("T0", "_T0", int, None, "rank 0 tensor"),
            "T1": Item("T1", "_T1", int, None, "rank 1 tensor"),
            "T2": Item("T2", "_T2", int, None, "rank 2 tensor"),
            "loss": Item("loss", "_loss", float, None, "loss"),
            "step": Item("step", "_step", int, None, "step"),
            "epoch": Item("epoch", "_epoch", int, None, "epoch")

        }}

    def __init__(self, scope_name: str = "default") -> None:
        if scope_name not in self._scopes:
            self._scopes[scope_name] = {}
        self._current = scope_name

    def __getattr__(self, alias: str):
    
        if alias in self._scopes:
            return Alias(alias)
        elif alias in self._current_scope:
            return self._current_scope[alias].key
        else:
            raise AttributeError(f"alias '{alias}' not found in {self._current} scope")
        
    def __getitem__(self, alias: str):

        if alias in self._current_scope:
            return self._current_scope[alias]
        else:
            raise KeyError(f"alias '{alias}' not found in {self._current} scope")

    def __getstate__(self) -> dict:
        return {
            "_scopes": self._scopes,
            "_current": "default"
        }
    
    def __call__(self, scope_name:str) -> 'Alias':
        if scope_name not in self._scopes:
            self._scopes[scope_name] = {}
        return self
    
    def __iter__(self):
        yield from self._current_scope
    
    def __contains__(self, alias: str) -> bool:
        return alias in self._current_scope

    def __setstate__(self, value: dict[str, Item]) -> None:
        self._scopes = value["_scopes"]
        self._current = value["_current"]

    def set(self, alias: str, keyword: str, type: Any, unit: str, comment: str) -> None:
        self._current_scope[alias] = Alias.Item(alias, keyword, type, unit, comment)

    def alias(self)->list[str]:
        return list(self._current_scope.keys())
    
    def items(self)->list[Item]:
        return list(self._current_scope.values())
    
    def get_unit(self, alias: str) -> str:
        return self._current_scope[alias].unit

    @property
    def _current_scope(self) -> str:
        return self._scopes[self._current]

    def map(self, alias, key):
        alias.key = key.key
        alias.alias = key.alias
        alias.type = key.type
        alias.unit = key.unit
        alias.comment = key.comment

alias = Alias()

# alias = Alias('default')
# alias.set('energy', 'energy', None, '')



# alias.set("cell", "_cell", None, "unit cell")
# kw.set("strain", "strain", None, "")
# alias.set("pbc", "_pbc", None, "periodic boundary conditions")
# kw.set("seg_m", "_seg_m", None, "start indices of systems")
# kw.set("idx_m", "_idx_m", None, "indices of systems")
# kw.set("idx_i", "_idx_i", None, "indices of center atoms")
# kw.set("idx_j", "_idx_j", None, "indices of neighboring atoms")
# kw.set(
#     "idx_i_lr", "_idx_i_lr", None, "indices of center atoms for long-range"
# )
# kw.set(
#     "idx_j_lr", "_idx_j_lr", None, "indices of neighboring atoms for long-range"
# )
# kw.set(
#     "lidx_i",
#     "_idx_i_local",
#     None,
#     "local indices of center atoms (within system)",
# )
# kw.set(
#     "lidx_j",
#     "_idx_j_local",
#     None,
#     "local indices of neighboring atoms (within system)",
# )
# alias.set(
#     "Rij",
#     "_Rij",
#     None,
#     "vectors pointing from center atoms to neighboring atoms",
# )
# kw.set(
#     "Rij_lr",
#     "_Rij_lr",
#     None,
#     "vectors pointing from center atoms to neighboring atoms for long range",
# )
# kw.set("n_atoms", "_n_atoms", None, "number of atoms")
# kw.set("offsets", "_offsets", None, "cell offset vectors")
# kw.set(
#     "offsets_lr", "_offsets_lr", None, "cell offset vectors for long range"
# )
# kw.set(
#     "R_strained",
#     "position_strained",
#     None,
#     "atom positions with strain-dependence",
# )
# kw.set(
#     "cell_strained",
#     "cell_strained",
#     None,
#     "atom positions with strain-dependence",
# )
# kw.set("n_nbh", "_n_nbh", None, "number of neighbors")
# kw.set(
#     "idx_i_triples", "_idx_i_triples", None, "indices of center atom triples"
# )
# kw.set(
#     "idx_j_triples",
#     "_idx_j_triples",
#     None,
#     "indices of first neighboring atom triples",
# )
# kw.set(
#     "idx_k_triples",
#     "_idx_k_triples",
#     None,
#     "indices of second neighboring atom triples",
# )
# kw.set("energy", "energy", None, "")
# kw.set("forces", "forces", None, "")
# kw.set("stress", "stress", None, "")
# kw.set("masses", "masses", None, "")
# kw.set("dipole_moment", "dipole_moment", None, "")
# kw.set("polarizability", "polarizability", None, "")
# kw.set("hessian", "hessian", None, "")
# kw.set("dipole_derivatives", "dipole_derivatives", None, "")
# kw.set(
#     "polarizability_derivatives", "polarizability_derivatives", None, ""
# )
# kw.set("total_charge", "total_charge", None, "")
# kw.set("partial_charges", "partial_charges", None, "")
# kw.set("spin_multiplicity", "spin_multiplicity", None, "")
# kw.set("electric_field", "electric_field", None, "")
# kw.set("magnetic_field", "magnetic_field", None, "")
# kw.set("nuclear_magnetic_moments", "nuclear_magnetic_moments", None, "")
# kw.set("shielding", "shielding", None, "")
# kw.set("nuclear_spin_coupling", "nuclear_spin_coupling", None, "")

# ## external fields needed for different response properties
# required_external_fields = {
#     kw.dipole_moment: [kw.electric_field],
#     kw.dipole_derivatives: [kw.electric_field],
#     kw.partial_charges: [kw.electric_field],
#     kw.polarizability: [kw.electric_field],
#     kw.polarizability_derivatives: [kw.electric_field],
#     kw.shielding: [kw.magnetic_field],
#     kw.nuclear_spin_coupling: [kw.magnetic_field],
# }

# __all__ = ["kw", "required_external_fields", "Keywords"]