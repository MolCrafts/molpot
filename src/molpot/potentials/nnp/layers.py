import torch
import torch.nn as nn


import numpy as np

__all__ = [
    "CutoffFunc",
    "GaussianBasis",
    "PolynomialBasis",
    "AtomicOnehot",
    "ANNOutput",
]


class AtomicOnehot(nn.Module):
    R"""One-hot embedding layer

    Given the atomic number of each atom ($Z_{i}$) and a list of specified
    element types denoted as ($Z^{\mathrm{0}}_{\alpha}$), returns:

    $$\mathbb{P}_{i\alpha} = \delta_{Z_{i},Z^{\mathrm{0}}_{\alpha}}$$
    """

    def __init__(self, atom_types=[1, 6, 7, 8, 9]):
        """
        Args:
            atom_types (list of int): list of elements ($Z^{0}$)
        """
        super(AtomicOnehot, self).__init__()
        self.atom_types = atom_types

    def call(self, elems):
        """
        Args:
           elems (tensor): atomic indices of atoms, with shape `(n_atoms)`

        Returns:
           prop (tensor): atomic property tensor, with shape `(n_atoms, n_elems)`
        """
        prop = torch.equal(
            torch.expand_dims(elems, 1), torch.expand_dims(self.atom_types, 0)
        )
        return prop


class ANNOutput(nn.Module):
    """ANN Ouput layer

    Output atomic or molecular (system) properties depending on `out_pool`

    $$
    \\begin{cases}
     \mathbb{P}^{\mathrm{out}}_i  &, \\textrm{if out_pool is False}\\\\
     \mathrm{pool}_i(\mathbb{P}^{\mathrm{out}}_i)  &, \\textrm{if out_pool}
    \end{cases}
    $$

    , where $\mathrm{pool}$ is a reducing operation specified with `out_pool`,
    it can be one of 'sum', 'max', 'min', 'avg'.

    """

    def __init__(self, out_pool):
        super(ANNOutput, self).__init__()
        self.out_pool = out_pool

    def call(self, tensors):
        """
        Args:
            tensors (list of tensor): ind_1 and output tensors

        Returns:
            output (tensor): atomic or per-structure predictions
        """
        ind_1, output = tensors

        if self.out_pool:
            out_pool = {
                "sum": torch.math.unsorted_segment_sum,
                "max": torch.math.unsorted_segment_max,
                "min": torch.math.unsorted_segment_min,
                "avg": torch.math.unsorted_segment_mean,
            }[self.out_pool]
            output = out_pool(output, ind_1[:, 0], torch.reduce_max(ind_1) + 1)
        output = torch.squeeze(output, axis=1)

        return output


class CutoffFunc(nn.Module):
    R"""Cutoff function layer

    The following types of cutoff function are implemented (all functions are
    defined within $r_{ij}<r_{c}$ and decay to zero at $r_{c}$):

    - $f^1(r_{ij}) = 0.5 (\mathrm{cos}(\pi r_{ij}/r_{c})+1)$
    - $f^2(r_{ij}) = \mathrm{tanh}^3(1- r_{ij}/r_{c})/\mathrm{tanh}^3(1)$
    - $hip(r_{ij}) = \mathrm{cos}^2(\pi r_{ij}/2r_{c})$

    """

    def __init__(self, rc=5.0, cutoff_type="f1"):
        """
        Args:
            rc (float): cutoff radius
            cutoff_type (string): name of the cutoff function
        """
        super(CutoffFunc, self).__init__()
        self.cutoff_type = cutoff_type
        self.rc = rc
        f1 = lambda x: 0.5 * (torch.cos(np.pi * x / rc) + 1)
        f2 = lambda x: (torch.tanh(1 - x / rc) / np.tanh(1)) ** 3
        hip = lambda x: torch.cos(np.pi * x / rc / 2) ** 2
        self.cutoff_fn = {"f1": f1, "f2": f2, "hip": hip}[cutoff_type]

    def forward(self, dist):
        """
        Args:
            dist (tensor): distance tensor with arbitrary shape

        Returns:
            fc (tensor): cutoff function with the same shape as the input
        """
        return self.cutoff_fn(dist)


class GaussianBasis(nn.Module):
    R"""Gaussian Basis Layer

    Builds the Gaussian basis function:

    $$
    e_{ijb} = e^{-\eta_b (r_{ij}-r_{b})^2}
    $$


    Both the Gaussian centers $r_{b}$ and width $\eta_{b}$ can be arrays that
    specifies the parameter for each basis function. When $\eta$ is given as a
    single float, the same value is assigned to every basis. When center is not
    given, `n_basis` and `rc` are used to generat a linearly spaced set of
    basis.

    """

    def __init__(self, center=None, gamma=None, rc=None, n_basis=None):
        """
        Args:
            center (float or array): Gaussian centers
            gamma (float or array): inverse Gaussian width
            rc (float): cutoff radius
            n_basis (int): number of basis function

        """
        super(GaussianBasis, self).__init__()
        if center is None:
            self.center = np.linspace(0, rc, n_basis)
        else:
            self.center = np.array(center)
        self.gamma = np.broadcast_to(gamma, self.center.shape)

    def forward(self, dist, fc=None):
        """
        Args:
           dist (tensor): distance tensor with shape (n_pairs)
           fc (tensor, optional): when supplied, apply a cutoff function to the basis

        Returns:
            basis (tensor): basis functions with shape (n_pairs, n_basis)
        """
        basis = torch.stack(
            [
                torch.exp(-gamma * (dist - center) ** 2)
                for (center, gamma) in zip(self.center, self.gamma)
            ],
            axis=1,
        )
        if fc is not None:
            basis = torch.einsum("pb,p->pb", basis, fc)  # p-> pair; b-> basis
        return basis


class PolynomialBasis(nn.Module):
    """Polynomial Basis Layer

    Builds the polynomial basis function:

    $$
    e_{ijb} = \mathrm{power}(r_{ij}, {n_{b}})
    $$

    , where $n_b$ is specified by `n_basis`. `n_basis` can be a list that
    explicitly specifies polynomail orders, or an integer that specifies a the
    orders as `[0, 1, ..., n_basis-1]`.

    """

    def __init__(self, n_basis):
        """

        Args:
            n_basis (int or list): number of basis function
        """
        super(PolynomialBasis, self).__init__()
        if type(n_basis) != list:
            n_basis = [(i + 1) for i in range(n_basis)]
        self.n_basis = n_basis

    def forward(self, dist, fc=None):
        """
        Args:
           dist (tensor): distance tensor with shape (n_pairs)
           fc (tensor): when supplied, apply a cutoff function to the basis

        Returns:
            basis (tensor): basis functions with shape (n_pairs, n_basis)
        """
        assert fc is not None, "Polynomail basis requires a cutoff function."
        basis = torch.stack([fc**i for i in self.n_basis], axis=1)
        return basis
