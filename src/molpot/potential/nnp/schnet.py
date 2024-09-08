from typing import Callable

import torch
from torch import nn
from molpot_op.scatter import scatter_add

import molpot as mpot
import molpot.potential.nnp as nnp

__all__ = ["SchNet", "SchNetInteraction"]


class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = nnp.shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = nnp.Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            nnp.Dense(n_filters, n_atom_basis, activation=activation),
            nnp.Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            nnp.Dense(n_rbf, n_filters, activation=activation), nnp.Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim=0, dim_size=x.shape[0])

        x = self.f2out(x)
        return x



class SchNet(nn.Module):
    """SchNet architecture for learning representations of atomistic systems

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        radial_basis: nn.Module,
        cutoff_fn: Callable,
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 101,
        activation: Callable | nn.Module = nnp.shifted_softplus,
        activate_charge_spin_embedding: bool = False,
        embedding: Callable | nn.Module = None,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
            activate_charge_spin_embedding: if True, charge and spin embeddings are added to nuclear embeddings taken from SpookyNet Implementation
            embedding: type of nuclear embedding to use (simple is simple embedding and complex is the one with electron configuration)
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.activate_charge_spin_embedding = activate_charge_spin_embedding

        self.alias = mpot.Alias('schnet')
        self.alias.set('scalar', '_schnet_scalar', torch.Tensor, None, 'scalar representation')

        # initialize nuclear embedding
        self.embedding = embedding
        if self.embedding is None:
            self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        # initialize spin and charge embeddings
        if self.activate_charge_spin_embedding:
            self.charge_embedding = nnp.ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=True)
            self.spin_embedding = nnp.ElectronicEmbedding(
                self.n_atom_basis,
                num_residual=1,
                activation=activation,
                is_charged=False)

        # initialize interaction blocks
        self.interactions = nnp.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: dict[str, torch.Tensor]):

        # get tensors from input dictionary
        atomic_numbers = inputs[mpot.Alias.Z]
        R = inputs[mpot.Alias.R]
        idx_i = inputs[mpot.Alias.idx_i]
        idx_j = inputs[mpot.Alias.idx_j]
        offsets = inputs[mpot.Alias.offsets]
        r_ij = R[idx_j] - R[idx_i] + offsets

        # r_ij = inputs[mpot.Alias.Rij]

        # compute pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute initial embeddings
        x = self.embedding(atomic_numbers)

        # add spin and charge embeddings
        if hasattr(self, "activate_charge_spin_embedding") and self.activate_charge_spin_embedding:
            # get tensors from input dictionary
            total_charge = inputs[mpot.Alias.total_charge]
            spin = inputs[mpot.Alias.spin_multiplicity]
            idx_m = inputs[mpot.Alias.idx_m]
            num_batch = len(inputs[mpot.Alias.idx])

            charge_embedding = self.charge_embedding(
                x, total_charge, num_batch, idx_m
            )
            spin_embedding = self.spin_embedding(
                x, spin, num_batch, idx_m
            )

            # additive combining of nuclear, charge and spin embedding
            x = x + charge_embedding + spin_embedding

        # compute interaction blocks and update atomic embeddings
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        # collect results
        inputs[mpot.Alias.schnet.scalar] = x

        return inputs
