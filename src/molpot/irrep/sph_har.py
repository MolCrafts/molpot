import torch
import torch_harmonics as th


def spherical_harmonics(
    l: Union[int, List[int], str, o3.Irreps], x: torch.Tensor, normalize: bool, normalization: str = "integral"
):
    r"""Spherical harmonics

    .. image:: https://user-images.githubusercontent.com/333780/79220728-dbe82c00-7e54-11ea-82c7-b3acbd9b2246.gif

    | Polynomials defined on the 3d space :math:`Y^l: \mathbb{R}^3 \longrightarrow \mathbb{R}^{2l+1}`
    | Usually restricted on the sphere (with ``normalize=True``) :math:`Y^l: S^2 \longrightarrow \mathbb{R}^{2l+1}`
    | who satisfies the following properties:

    * are polynomials of the cartesian coordinates ``x, y, z``
    * is equivariant :math:`Y^l(R x) = D^l(R) Y^l(x)`
    * are orthogonal :math:`\int_{S^2} Y^l_m(x) Y^j_n(x) dx = \text{cste} \; \delta_{lj} \delta_{mn}`

    The value of the constant depends on the choice of normalization.

    It obeys the following property:

    .. math::

        Y^{l+1}_i(x) &= \text{cste}(l) \; & C_{ijk} Y^l_j(x) x_k

        \partial_k Y^{l+1}_i(x) &= \text{cste}(l) \; (l+1) & C_{ijk} Y^l_j(x)

    Where :math:`C` are the `wigner_3j`.

    .. note::

        This function match with this table of standard real spherical harmonics from Wikipedia_
        when ``normalize=True``, ``normalization='integral'`` and is called with the argument in the order ``y,z,x``
        (instead of ``x,y,z``).

    .. _Wikipedia: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

    Parameters
    ----------
    l : int or list of int
        degree of the spherical harmonics.

    x : `torch.Tensor`
        tensor :math:`x` of shape ``(..., 3)``.

    normalize : bool
        whether to normalize the ``x`` to unit vectors that lie on the sphere before projecting onto the spherical harmonics

    normalization : {'integral', 'component', 'norm'}
        normalization of the output tensors --- note that this option is independent of ``normalize``, which controls the
        processing of the *input*, rather than the output.
        Valid options:
        * *component*: :math:`\|Y^l(x)\|^2 = 2l+1, x \in S^2`
        * *norm*: :math:`\|Y^l(x)\| = 1, x \in S^2`, ``component / sqrt(2l+1)``
        * *integral*: :math:`\int_{S^2} Y^l_m(x)^2 dx = 1`, ``component / sqrt(4pi)``

    Returns
    -------
    `torch.Tensor`
        a tensor of shape ``(..., 2l+1)``

        .. math:: Y^l(x)

    Examples
    --------

    >>> spherical_harmonics(0, torch.randn(2, 3), False, normalization='component')
    tensor([[1.],
            [1.]])

    See Also
    --------
    wigner_D
    wigner_3j

    """
    sh = th
    sh = SphericalHarmonics(l, normalize, normalization)
    return sh(x)