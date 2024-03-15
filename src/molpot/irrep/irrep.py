
import torch


class Irrep(tuple):

    def __new__(cls, l, p:int=0):

        if p == 0:
            if isinstance(l, Irrep):
                p = l.p
                l = l.l

            # if isinstance(l, MulIrrep):
            #     p = l.ir.p
            #     l = l.ir.l

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

        assert p in [-1, 0, 1], "p must be -1, 0, or 1"

        return super().__new__(cls, (l, p))
    
    @property
    def l(self):
        return self[0]
    
    @property
    def p(self):
        return self[1]
    
    def __repr__(self):
        """Representation of the Irrep."""
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"
    
    def D_from_angles(self, alpha, beta, gamma, k=0):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`.

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`.

        Args:
            alpha (`jax.Array`): of shape :math:`(...)`
                Rotation :math:`\alpha` around Y axis, applied third.
            beta (`jax.Array`): of shape :math:`(...)`
                Rotation :math:`\beta` around X axis, applied second.
            gamma (`jax.Array`): of shape :math:`(...)`
                Rotation :math:`\gamma` around Y axis, applied first.
            k (optional `jax.Array`): of shape :math:`(...)`
                How many times the parity is applied.

        Returns:
            `jax.Array`: of shape :math:`(..., 2l+1, 2l+1)`

        See Also:
            Irreps.D_from_angles
        """
        k = torch.asarray(k)
        if isinstance(alpha, (int, float)) and alpha == 0:
            alpha = None
        else:
            alpha = torch.asarray(alpha)

        if isinstance(beta, (int, float)) and beta == 0:
            beta = None
        else:
            beta = torch.asarray(beta)

        if isinstance(gamma, (int, float)) and gamma == 0:
            gamma = None
        else:
            gamma = torch.asarray(gamma)

        shape = torch.broadcast_shapes(
            *[a.shape for a in [alpha, beta, gamma] if a is not None], k.shape
        )

        if alpha is not None:
            alpha = torch.broadcast_to(alpha, shape)
        if beta is not None:
            beta = torch.broadcast_to(beta, shape)
        if gamma is not None:
            gamma = torch.broadcast_to(gamma, shape)
        k = torch.broadcast_to(k, shape)

        return (
            _wigner_D_from_angles(self.l, alpha, beta, gamma)
            * self.p ** k[..., None, None]
        )

    # def D_from_matrix(self, R):
    #     r"""Matrix of the representation.

    #     Args:
    #         R (`jax.Array`): array of shape :math:`(..., 3, 3)`
    #         k (`jax.Array`, optional): array of shape :math:`(...)`

    #     Returns:
    #         `jax.Array`: array of shape :math:`(..., 2l+1, 2l+1)`

    #     Examples:
    #         >>> m = Irrep(1, -1).D_from_matrix(-jnp.eye(3))
    #         >>> m + 0.0
    #         Array([[-1.,  0.,  0.],
    #                [ 0., -1.,  0.],
    #                [ 0.,  0., -1.]], dtype=float32)

    #     See Also:
    #         `Irrep.D_from_angles`
    #     """
    #     d = jnp.sign(jnp.linalg.det(R))
    #     R = d[..., None, None] * R
    #     k = (1 - d) / 2
    #     return self.D_from_angles(*matrix_to_angles(R), k)
