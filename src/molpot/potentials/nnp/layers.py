import torch
import torch.nn as nn
from typing import Callable, Union, Sequence
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_

from molpot import Config


def build_mlp(
    n_neurons: Union[int, Sequence[int]],
    activation: Union[Callable, nn.Module] | None = None,
    last_bias: bool = True,
    last_zero_init: bool = False,
    **kwargs
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers

    n_layers = len(n_neurons)

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation, **kwargs)
        for i in range(n_layers - 2)
    ]
    # assign a Dense layer (without activation function) to the output layer
    if last_zero_init:
        layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                **kwargs
            )
        )
    else:
        layers.append(Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias))
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias, device=Config.device, dtype=Config.ftype)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff], device=Config.device))

    def forward(self, input: torch.Tensor):
        # Compute values of cutoff function
        input_cut = 0.5 * (torch.cos(input * torch.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        input_cut *= (input < self.cutoff).float()
        return input_cut


class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_basis: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        r"""
        Args:
            n_basis: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_basis = n_basis

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_basis)
        widths = torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        offset = offset.to(Config.device)
        widths = widths.to(Config.device)

        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, dij: torch.Tensor, fc: torch.Tensor):
        coeff = -0.5 / torch.pow(self.widths, 2)
        diff = dij[..., None] - self.offsets
        y = torch.exp(coeff * torch.pow(diff, 2))
        basis = y * fc[..., None]
        return basis


class AtomicOnehot(nn.Module):
    r"""One-hot embedding layer

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
    r"""ANN Ouput layer

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


class PolynomialBasis(nn.Module):
    r"""Polynomial Basis Layer

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
