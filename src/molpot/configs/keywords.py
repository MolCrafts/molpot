"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from typing import Final

class Keywords:

    def __init__(self, name="_mpot_global"):
        self._name = name
        self._maps:dict[str, str] = {}
        self._units:dict[str, str] = {}
        self._comments:dict[str, str] = {}

    def __getitem__(self, key):
        return self._maps[key]
    
    def __setitem__(self, key, value):
        self._maps[key] = value

    def __getattribute__(self, __name: str) -> str:
        return self._maps[__name]
    
    def __setattr__(self, __name: str, value: str) -> None:
        self._maps[__name] = value

    def get_unit(self, key):
        return self._units[key]
    
    def set_unit(self, key, value):
        self._units[key] = value

    def set(self, alias, keyword, unit=None, comment=None):
        self._maps[alias] = keyword
        self._units[alias] = unit
        self._comments[alias] = comment
        self.__annotations__[alias] = Final[str]
        self.__doc__ += f"\n    {alias}: {keyword} ({unit}) # {comment}"

keywords = Keywords()
keywords.set("idx", "_idx", None, "")
keywords.set("Z", "_atomic_numbers", None, "nuclear charge")
keywords.set("position", "_positions", None, "atom positions")
keywords.set("R", "position", None, "atom positions")
keywords.set("cell", "_cell", None, "unit cell")
keywords.set("strain", "strain", None, "")
keywords.set("pbc", "_pbc", None, "periodic boundary conditions")
keywords.set("seg_m", "_seg_m", None, "start indices of systems")
keywords.set("idx_m", "_idx_m", None, "indices of systems")
keywords.set("idx_i", "_idx_i", None, "indices of center atoms")
keywords.set("idx_j", "_idx_j", None, "indices of neighboring atoms")
keywords.set("idx_i_lr", "_idx_i_lr", None, "indices of center atoms for long-range")
keywords.set("idx_j_lr", "_idx_j_lr", None, "indices of neighboring atoms for long-range")
keywords.set("lidx_i", "_idx_i_local", None, "local indices of center atoms (within system)")
keywords.set("lidx_j", "_idx_j_local", None, "local indices of neighboring atoms (within system)")
keywords.set("Rij", "_Rij", None, "vectors pointing from center atoms to neighboring atoms")
keywords.set("Rij_lr", "_Rij_lr", None, "vectors pointing from center atoms to neighboring atoms for long range")
keywords.set("n_atoms", "_n_atoms", None, "number of atoms")
keywords.set("offsets", "_offsets", None, "cell offset vectors")
keywords.set("offsets_lr", "_offsets_lr", None, "cell offset vectors for long range")
keywords.set("R_strained", "position_strained", None, "atom positions with strain-dependence")
keywords.set("cell_strained", "cell_strained", None, "atom positions with strain-dependence")
keywords.set("n_nbh", "_n_nbh", None, "number of neighbors")
keywords.set("idx_i_triples", "_idx_i_triples", None, "indices of center atom triples")
keywords.set("idx_j_triples", "_idx_j_triples", None, "indices of first neighboring atom triples")
keywords.set("idx_k_triples", "_idx_k_triples", None, "indices of second neighboring atom triples")
keywords.set("energy", "energy", None, "")
keywords.set("forces", "forces", None, "")
keywords.set("stress", "stress", None, "")
keywords.set("masses", "masses", None, "")
keywords.set("dipole_moment", "dipole_moment", None, "")
keywords.set("polarizability", "polarizability", None, "")
keywords.set("hessian", "hessian", None, "")
keywords.set("dipole_derivatives", "dipole_derivatives", None, "")
keywords.set("polarizability_derivatives", "polarizability_derivatives", None, "")
keywords.set("total_charge", "total_charge", None, "")
keywords.set("partial_charges", "partial_charges", None, "")
keywords.set("spin_multiplicity", "spin_multiplicity", None, "")
keywords.set("electric_field", "electric_field", None, "")
keywords.set("magnetic_field", "magnetic_field", None, "")
keywords.set("nuclear_magnetic_moments", "nuclear_magnetic_moments", None, "")
keywords.set("shielding", "shielding", None, "")
keywords.set("nuclear_spin_coupling", "nuclear_spin_coupling", None, "")

## external fields needed for different response properties
required_external_fields = {
    keywords.dipole_moment: [keywords.electric_field],
    keywords.dipole_derivatives: [keywords.electric_field],
    keywords.partial_charges: [keywords.electric_field],
    keywords.polarizability: [keywords.electric_field],
    keywords.polarizability_derivatives: [keywords.electric_field],
    keywords.shielding: [keywords.magnetic_field],
    keywords.nuclear_spin_coupling: [keywords.magnetic_field],
}

def def_new_keywords(name:str) -> Keywords:
    return Keywords(name)