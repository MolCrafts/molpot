import collections
import itertools
from typing import List, Union

import torch

# These imports avoid cyclic reference from o3 itself
from . import perm, rotation, wigner
from .linalg import direct_sum


class Irrep(tuple):
    r"""Irreducible representation of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of
    functions.

    Parameters
    ----------
    l : int
        non-negative integer, the degree of the representation, :math:`l = 0, 1, \dots`

    p : {1, -1}
        the parity of the representation

    Examples
    --------
    Create a scalar representation (:math:`l=0`) of even parity.

    >>> Irrep(0, 1)
    0e

    Create a pseudotensor representation (:math:`l=2`) of odd parity.

    >>> Irrep(2, -1)
    2o

    Create a vector representation (:math:`l=1`) of the parity of the spherical harmonics (:math:`-1^l` gives odd parity).

    >>> Irrep("1y")
    1o

    >>> Irrep("2o").dim
    5

    >>> Irrep("2e") in Irrep("1o") * Irrep("1o")
    True

    >>> Irrep("1o") + Irrep("2o")
    1x1o+1x2o
    """

    def __new__(cls, l: Union[int, "Irrep", str, tuple], p=None):
        if p is None:
            if isinstance(l, Irrep):
                return l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    assert l >= 0
                    p = {
                        "e": 1,
                        "o": -1,
                        "y": (-1) ** l,
                    }[name[-1]]
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                l, p = l

        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be positive integer, got {l}")
        if p not in (-1, 1):
            raise ValueError(f"parity must be on of (-1, 1), got {p}")
        return super().__new__(cls, (l, p))

    @property
    def l(self) -> int:  # noqa: E743
        r"""The degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def p(self) -> int:
        r"""The parity of the representation, :math:`p = \pm 1`."""
        return self[1]

    def __repr__(self) -> str:
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"

    @classmethod
    def iterator(cls, lmax=None):
        r"""Iterator through all the irreps of :math:`O(3)`

        Examples
        --------
        >>> it = Irrep.iterator()
        >>> next(it), next(it), next(it), next(it)
        (0e, 0o, 1o, 1e)
        """
        for l in itertools.count():
            yield Irrep(l, (-1) ** l)
            yield Irrep(l, -((-1) ** l))

            if l == lmax:
                break

    def D_from_angles(self, alpha, beta, gamma, k=None) -> torch.Tensor:
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\alpha` around Y axis, applied third.

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\beta` around X axis, applied second.

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\gamma` around Y axis, applied first.

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the parity is applied.

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        See Also
        --------
        o3.wigner_D
        Irreps.D_from_angles
        """
        if k is None:
            k = torch.zeros_like(alpha)

        alpha, beta, gamma, k = torch.broadcast_tensors(alpha, beta, gamma, k)
        return wigner.wigner_D(self.l, alpha, beta, gamma) * self.p ** k[..., None, None]

    def D_from_quaternion(self, q, k=None) -> torch.Tensor:
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*rotation.quaternion_to_angles(q), k)

    def D_from_matrix(self, R) -> torch.Tensor:
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        Examples
        --------
        >>> m = Irrep(1, -1).D_from_matrix(-torch.eye(3))
        >>> m.long()
        tensor([[-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1]])
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*rotation.matrix_to_angles(R), k)

    def D_from_axis_angle(self, axis, angle) -> torch.Tensor:
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        axis : `torch.Tensor`
            tensor of shape :math:`(..., 3)`

        angle : `torch.Tensor`
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*rotation.axis_angle_to_angles(axis, angle))

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1``"""
        return self.l == 0 and self.p == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns
        -------
        generator of `e3nn.o3.Irrep`
        """
        other = Irrep(other)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1e')
        3x1e
        """
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self) -> str:
        return f"{self.mul}x{self.ir}"

    def __getitem__(self, item) -> Union[int, Irrep]:  # pylint: disable=useless-super-delegation
        return super().__getitem__(item)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


class Irreps(tuple):
    r"""Direct sum of irreducible representations of :math:`O(3)`

    This class does not contain any data, it is a structure that describe the representation.
    It is typically used as argument of other classes of the library to define the input and output representations of
    functions.

    Attributes
    ----------
    dim : int
        the total dimension of the representation

    num_irreps : int
        number of irreps. the sum of the multiplicities

    ls : list of int
        list of :math:`l` values

    lmax : int
        maximum :math:`l` value

    Examples
    --------
    Create a representation of 100 :math:`l=0` of even parity and 50 pseudo-vectors.

    >>> x = Irreps([(100, (0, 1)), (50, (1, 1))])
    >>> x
    100x0e+50x1e

    >>> x.dim
    250

    Create a representation of 100 :math:`l=0` of even parity and 50 pseudo-vectors.

    >>> Irreps("100x0e + 50x1e")
    100x0e+50x1e

    >>> Irreps("100x0e + 50x1e + 0x2e")
    100x0e+50x1e+0x2e

    >>> Irreps("100x0e + 50x1e + 0x2e").lmax
    1

    >>> Irrep("2e") in Irreps("0e + 2e")
    True

    Empty Irreps

    >>> Irreps(), Irreps("")
    (, )
    """

    def __new__(cls, irreps=None) -> Union[_MulIr, "Irreps"]:
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1) -> "Irreps":
        r"""representation of the spherical harmonics

        Parameters
        ----------
        lmax : int
            maximum :math:`l`

        p : {1, -1}
            the parity of the representation

        Returns
        -------
        `e3nn.o3.Irreps`
            representation of :math:`(Y^0, Y^1, \dots, Y^{\mathrm{lmax}})`

        Examples
        --------

        >>> Irreps.spherical_harmonics(3)
        1x0e+1x1o+1x2e+1x3o

        >>> Irreps.spherical_harmonics(4, p=1)
        1x0e+1x1e+1x2e+1x3e+1x4e
        """
        return Irreps([(1, (l, p**l)) for l in range(lmax + 1)])

    def slices(self):
        r"""List of slices corresponding to indices for each irrep.

        Examples
        --------

        >>> Irreps('2x0e + 1e').slices()
        [slice(0, 2, None), slice(2, 5, None)]
        """
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(
        self, *size: int, normalization: str = "component", requires_grad: bool = False, dtype=None, device=None
    ) -> torch.Tensor:
        r"""Random tensor.

        Parameters
        ----------
        *size : list of int
            size of the output tensor, needs to contains a ``-1``

        normalization : {'component', 'norm'}

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``size`` where ``-1`` is replaced by ``self.dim``

        Examples
        --------

        >>> Irreps("5x0e + 10x1o").randn(5, -1, 5, normalization='norm').shape
        torch.Size([5, 35, 5])

        >>> random_tensor = Irreps("2o").randn(2, -1, 3, normalization='norm')
        >>> random_tensor.norm(dim=1).sub(1).abs().max().item() < 1e-5
        True
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]

        if normalization == "component":
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, "Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.

        Parameters
        ----------
        ir : `e3nn.o3.Irrep`

        Returns
        -------
        `int`
            total multiplicity of ``ir``
        """
        ir = Irrep(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps) -> "Irreps":
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other) -> "Irreps":
        r"""
        >>> (Irreps('2x1e') * 3).simplify()
        6x1e
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other) -> "Irreps":
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
        return Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        """Simplify the representations.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        Note that simplify does not sort the representations.

        >>> Irreps("1e + 1e + 0e").simplify()
        2x1e+1x0e

        Equivalent representations which are separated from each other are not combined.

        >>> Irreps("1e + 1e + 0e + 1e").simplify()
        2x1e+1x0e+1x1e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self) -> "Irreps":
        """Remove any irreps with multiplicities of zero.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
        4x0e+2x3e

        """
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return Irreps(out)

    def sort(self):
        r"""Sort the representations.

        Returns
        -------
        irreps : `e3nn.o3.Irreps`
        p : tuple of int
        inv : tuple of int

        Examples
        --------

        >>> Irreps("1e + 0e + 1e").sort().irreps
        1x0e+1x1e+1x1e

        >>> Irreps("2o + 1e + 0e + 1e").sort().p
        (3, 1, 0, 2)

        >>> Irreps("2o + 1e + 0e + 1e").sort().inv
        (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        irreps = Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, inv)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self) -> str:
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return direct_sum(*[ir.D_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    def D_from_quaternion(self, q, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*rotation.quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*rotation.matrix_to_angles(R), k)

    def D_from_axis_angle(self, axis, angle):
        r"""Matrix of the representation

        Parameters
        ----------
        axis : `torch.Tensor`
            tensor of shape :math:`(..., 3)`

        angle : `torch.Tensor`
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*rotation.axis_angle_to_angles(axis, angle))
    

class SphericalTensor(Irreps):
    r"""representation of a signal on the sphere

    A `SphericalTensor` contains the coefficients :math:`A^l` of a function :math:`f` defined on the sphere

    .. math::
        f(x) = \sum_{l=0}^{l_\mathrm{max}} A^l \cdot Y^l(x)


    The way this function is transformed by parity :math:`f \longrightarrow P f` is described by the two parameters :math:`p_v`
    and :math:`p_a`

    .. math::
        (P f)(x) &= p_v f(p_a x)

        &= \sum_{l=0}^{l_\mathrm{max}} p_v p_a^l A^l \cdot Y^l(x)


    Parameters
    ----------
    lmax : int
        :math:`l_\mathrm{max}`

    p_val : {+1, -1}
        :math:`p_v`

    p_arg : {+1, -1}
        :math:`p_a`


    Examples
    --------

    >>> SphericalTensor(3, 1, 1)
    1x0e+1x1e+1x2e+1x3e

    >>> SphericalTensor(3, 1, -1)
    1x0e+1x1o+1x2e+1x3o
    """
    # pylint: disable=abstract-method

    def __new__(
        # pylint: disable=signature-differs
        cls,
        lmax,
        p_val,
        p_arg,
    ):
        return super().__new__(cls, [(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])

    def with_peaks_at(self, vectors, values=None):
        r"""Create a spherical tensor with peaks

        The peaks are located in :math:`\vec r_i` and have amplitude :math:`\|\vec r_i \|`

        Parameters
        ----------
        vectors : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(N, 3)``

        values : `torch.Tensor`, optional
            value on the peak, tensor of shape ``(N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(self.dim,)``

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = torch.tensor([
        ...     [1.0, 0.0, 0.0],
        ...     [3.0, 4.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> s.signal_xyz(x, pos).long()
        tensor([1, 5])

        >>> val = torch.tensor([
        ...     -1.5,
        ...     2.0,
        ... ])
        >>> x = s.with_peaks_at(pos, val)
        >>> s.signal_xyz(x, pos)
        tensor([-1.5000,  2.0000])
        """
        if values is not None:
            vectors, values = torch.broadcast_tensors(vectors, values[..., None])
            values = values[..., 0]

        # empty set of vectors returns a 0 spherical tensor
        if vectors.numel() == 0:
            return torch.zeros(vectors.shape[:-2] + (self.dim,))

        assert (
            self[0][1].p == 1
        ), "since the value is set by the radii who is even, p_val has to be 1"  # pylint: disable=no-member

        assert vectors.dim() == 2 and vectors.shape[1] == 3

        if values is None:
            values = vectors.norm(dim=1)  # [batch]
        vectors = vectors[values != 0]  # [batch, 3]
        values = values[values != 0]

        coeff = o3.spherical_harmonics(self, vectors, normalize=True)  # [batch, l * m]
        A = torch.einsum("ai,bi->ab", coeff, coeff)
        # Y(v_a) . Y(v_b) solution_b = radii_a
        solution = torch.linalg.lstsq(A, values).solution.reshape(-1)  # [b]
        assert (values - A @ solution).abs().max() < 1e-5 * values.abs().max()

        return solution @ coeff

    def sum_of_diracs(self, positions: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        r"""Sum (almost-) dirac deltas

        .. math::

            f(x) = \sum_i v_i \delta^L(\vec r_i)

        where :math:`\delta^L` is the apporximation of a dirac delta.

        Parameters
        ----------
        positions : `torch.Tensor`
            :math:`\vec r_i` tensor of shape ``(..., N, 3)``

        values : `torch.Tensor`
            :math:`v_i` tensor of shape ``(..., N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(7, 1, -1)
        >>> pos = torch.tensor([
        ...     [1.0, 0.0, 0.0],
        ...     [0.0, 1.0, 0.0],
        ... ])
        >>> val = torch.tensor([
        ...     -1.0,
        ...     1.0,
        ... ])
        >>> x = s.sum_of_diracs(pos, val)
        >>> s.signal_xyz(x, torch.eye(3)).mul(10.0).round()
        tensor([-10.,  10.,  -0.])

        >>> s.sum_of_diracs(torch.empty(1, 0, 2, 3), torch.empty(2, 0, 1)).shape
        torch.Size([2, 0, 64])

        >>> s.sum_of_diracs(torch.randn(1, 3, 2, 3), torch.randn(2, 1, 1)).shape
        torch.Size([2, 3, 64])
        """
        positions, values = torch.broadcast_tensors(positions, values[..., None])
        values = values[..., 0]

        if positions.numel() == 0:
            return torch.zeros(values.shape[:-1] + (self.dim,))

        y = o3.spherical_harmonics(self, positions, True)  # [..., N, dim]
        v = values[..., None]

        return 4 * pi / (self.lmax + 1) ** 2 * (y * v).sum(-2)

    def from_samples_on_s2(self, positions: torch.Tensor, values: torch.Tensor, res: int = 100) -> torch.Tensor:
        r"""Convert a set of position on the sphere and values into a spherical tensor

        Parameters
        ----------
        positions : `torch.Tensor`
            tensor of shape ``(..., N, 3)``

        values : `torch.Tensor`
            tensor of shape ``(..., N)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., self.dim)``

        Examples
        --------
        >>> s = SphericalTensor(2, 1, 1)
        >>> pos = torch.tensor([
        ...     [
        ...         [0.0, 0.0, 1.0],
        ...         [0.0, 0.0, -1.0],
        ...     ],
        ...     [
        ...         [0.0, 1.0, 0.0],
        ...         [0.0, -1.0, 0.0],
        ...     ],
        ... ], dtype=torch.float64)
        >>> val = torch.tensor([
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ...     [
        ...         1.0,
        ...         -1.0,
        ...     ],
        ... ], dtype=torch.float64)
        >>> s.from_samples_on_s2(pos, val, res=200).long()
        tensor([[0, 0, 0, 3, 0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0, 0, 0, 0, 0]])

        >>> pos = torch.empty(2, 0, 10, 3)
        >>> val = torch.empty(2, 0, 10)
        >>> s.from_samples_on_s2(pos, val)
        tensor([], size=(2, 0, 9))

        """
        positions, values = torch.broadcast_tensors(positions, values[..., None])
        values = values[..., 0]

        if positions.numel() == 0:
            return torch.zeros(values.shape[:-1] + (self.dim,))

        positions = torch.nn.functional.normalize(positions, dim=-1)  # forward 0's instead of nan for zero-radius

        size = positions.shape[:-2]
        n = positions.shape[-2]
        positions = positions.reshape(-1, n, 3)
        values = values.reshape(-1, n)

        s2 = FromS2Grid(res=res, lmax=self.lmax, normalization="integral", dtype=values.dtype, device=values.device)
        pos = s2.grid.reshape(1, -1, 3)

        cd = torch.cdist(pos, positions)  # [batch, b*a, N]
        i = torch.arange(len(values)).view(-1, 1)  # [batch, 1]
        j = cd.argmin(2)  # [batch, b*a]
        val = values[i, j]  # [batch, b*a]
        val = val.reshape(*size, s2.res_beta, s2.res_alpha)

        return s2(val)

    def norms(self, signal) -> torch.Tensor:
        r"""The norms of each l component

        Parameters
        ----------
        signal : `torch.Tensor`
            tensor of shape ``(..., dim)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., lmax+1)``

        Examples
        --------
        Examples
        --------
        >>> s = SphericalTensor(1, 1, -1)
        >>> s.norms(torch.tensor([1.5, 0.0, 3.0, 4.0]))
        tensor([1.5000, 5.0000])
        """
        i = 0
        norms = []
        for _, ir in self:
            norms += [signal[..., i : i + ir.dim].norm(dim=-1)]
            i += ir.dim
        return torch.stack(norms, dim=-1)

    def signal_xyz(self, signal, r) -> torch.Tensor:
        r"""Evaluate the signal on given points on the sphere

        .. math::

            f(\vec x / \|\vec x\|)

        Parameters
        ----------
        signal : `torch.Tensor`
            tensor of shape ``(*A, self.dim)``

        r : `torch.Tensor`
            tensor of shape ``(*B, 3)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(*A, *B)``

        Examples
        --------
        >>> s = SphericalTensor(3, 1, -1)
        >>> s.signal_xyz(s.randn(2, 1, 3, -1), torch.randn(2, 4, 3)).shape
        torch.Size([2, 1, 3, 2, 4])
        """
        sh = o3.spherical_harmonics(self, r, normalize=True)
        dim = (self.lmax + 1) ** 2
        output = torch.einsum("bi,ai->ab", sh.reshape(-1, dim), signal.reshape(-1, dim))
        return output.reshape(signal.shape[:-1] + r.shape[:-1])

    def signal_on_grid(self, signal, res: int = 100, normalization: str = "integral"):
        r"""Evaluate the signal on a grid on the sphere"""
        Ret = namedtuple("Return", "grid, values")
        s2 = ToS2Grid(lmax=self.lmax, res=res, normalization=normalization)
        return Ret(s2.grid, s2(signal))

    def plotly_surface(
        self, signals, centers=None, res: int = 100, radius: bool = True, relu: bool = False, normalization: str = "integral"
    ):
        r"""Create traces for plotly

        Examples
        --------
        >>> import plotly.graph_objects as go
        >>> x = SphericalTensor(4, +1, +1)
        >>> traces = x.plotly_surface(x.randn(-1))
        >>> traces = [go.Surface(**d) for d in traces]
        >>> fig = go.Figure(data=traces)
        """
        signals = signals.reshape(-1, self.dim)

        if centers is None:
            centers = [None] * len(signals)
        else:
            centers = centers.reshape(-1, 3)

        traces = []
        for signal, center in zip(signals, centers):
            r, f = self.plot(signal, center, res, radius, relu, normalization)
            traces += [
                {
                    "x": r[:, :, 0].numpy(),
                    "y": r[:, :, 1].numpy(),
                    "z": r[:, :, 2].numpy(),
                    "surfacecolor": f.numpy(),
                }
            ]
        return traces

    def plot(
        self, signal, center=None, res: int = 100, radius: bool = True, relu: bool = False, normalization: str = "integral"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Create surface in order to make a plot"""
        assert signal.dim() == 1

        r, f = self.signal_on_grid(signal, res, normalization)
        f = f.relu() if relu else f

        # beta: [0, pi]
        r[0] = r.new_tensor([0.0, 1.0, 0.0])
        r[-1] = r.new_tensor([0.0, -1.0, 0.0])
        f[0] = f[0].mean()
        f[-1] = f[-1].mean()

        # alpha: [0, 2pi]
        r = torch.cat([r, r[:, :1]], dim=1)  # [beta, alpha, 3]
        f = torch.cat([f, f[:, :1]], dim=1)  # [beta, alpha]

        if radius:
            r *= f.abs().unsqueeze(-1)

        if center is not None:
            r += center

        return r, f

    def find_peaks(self, signal, res: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Locate peaks on the sphere

        Examples
        --------
        >>> s = SphericalTensor(4, 1, -1)
        >>> pos = torch.tensor([
        ...     [4.0, 0.0, 4.0],
        ...     [0.0, 5.0, 0.0],
        ... ])
        >>> x = s.with_peaks_at(pos)
        >>> pos, val = s.find_peaks(x)
        >>> pos[val > 4.0].mul(10).round().abs()
        tensor([[ 7.,  0.,  7.],
                [ 0., 10.,  0.]])
        >>> val[val > 4.0].mul(10).round().abs()
        tensor([57., 50.])
        """
        x1, f1 = self.signal_on_grid(signal, res)

        abc = torch.tensor([pi / 2, pi / 2, pi / 2])
        R = o3.angles_to_matrix(*abc)
        D = self.D_from_matrix(R)

        r_signal = D @ signal
        rx2, f2 = self.signal_on_grid(r_signal, res)
        x2 = torch.einsum("ij,baj->bai", R.T, rx2)

        ij = _find_peaks_2d(f1)
        x1p = torch.stack([x1[i, j] for i, j in ij])
        f1p = torch.stack([f1[i, j] for i, j in ij])

        ij = _find_peaks_2d(f2)
        x2p = torch.stack([x2[i, j] for i, j in ij])
        f2p = torch.stack([f2[i, j] for i, j in ij])

        # Union of the results
        mask = torch.cdist(x1p, x2p) < 2 * pi / res
        x = torch.cat([x1p[mask.sum(1) == 0], x2p])
        f = torch.cat([f1p[mask.sum(1) == 0], f2p])

        return x, f