"""
Keys to access structure properties.

Note: Had to be moved out of Structure class for TorchScript compatibility

"""
from molpy import Aliases

alias = Aliases('default')
alias.set('idx', '_idx', None, '')
alias.set('idx_m', '_idx_m', None, 'indices of systems')
alias.set('idx_i', '_idx_i', None, 'indices of center atoms')
alias.set('idx_j', '_idx_j', None, 'indices of neighboring atoms')
alias.set('idx_i_lr', '_idx_i_lr', None, 'indices of center atoms for long-range')
alias.set('idx_j_lr', '_idx_j_lr', None, 'indices of neighboring atoms for long-range')
alias.set('offsets', '_offsets', None, 'cell offset vectors')
alias.set('energy', 'energy', None, '')

# kw.set("idx", "_idx", None, "")
# kw.set("Z", "_atomic_numbers", None, "nuclear charge")
alias.set("cell", "_cell", None, "unit cell")
# kw.set("strain", "strain", None, "")
alias.set("pbc", "_pbc", None, "periodic boundary conditions")
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
alias.set(
    "Rij",
    "_Rij",
    None,
    "vectors pointing from center atoms to neighboring atoms",
)
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