import torch
import torch.nn as nn
from enum import IntEnum


class Box:

    class Style(IntEnum):
        FREE = 0
        ORTHOGONAL = 1
        TRICLINIC = 2

    def __init__(
        self,
        matrix: float | torch.Tensor | None = None,
        pbc: torch.Tensor = torch.zeros(3, dtype=bool),
    ):
        if matrix is None:
            self._matrix = 0
        elif isinstance(matrix, (int, float)):
            self._matrix = torch.eye(3) * matrix
        else:
            self._matrix = Box.check_matrix(matrix)
        self._pbc = pbc
        self._style = self.calc_style_from_matrix(self._matrix)

    def __repr__(self):
        match self.style:
            case Box.Style.FREE:
                return f"<Box: Free>"
            case Box.Style.ORTHOGONAL:
                return f"<Box: Orthogonal: {self.lengths}>"
            case Box.Style.TRICLINIC:
                return f"<Box: Triclinic: {self._matrix}>"

    @property
    def style(self) -> Style:
        return self.calc_style_from_matrix(self._matrix)

    @property
    def pbc(self) -> torch.Tensor:
        return self._pbc

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix

    @staticmethod
    def check_matrix(matrix: torch.Tensor) -> torch.Tensor:
        # assert isinstance(matrix, torch.Tensor), "matrix must be torch.Tensor"
        assert matrix.shape == (3, 3), "matrix must be (3, 3)"
        assert torch.det(matrix) != 0, "matrix must be non-singular"
        return matrix

    @property
    def lengths(self) -> torch.Tensor:
        match self.style:
            case Box.Style.FREE:
                return torch.zeros(3)
            case Box.Style.ORTHOGONAL | Box.Style.TRICLINIC:
                return self.calc_lengths_from_matrix(self._matrix)

    @property
    def angles(self) -> torch.Tensor:
        return self.calc_angles_from_matrix(self._matrix)

    @staticmethod
    def general2restrict(matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert general triclinc box matrix to restricted triclinic box matrix

        Ref:
            https://docs.lammps.org/Howto_triclinic.html#transformation-from-general-to-restricted-triclinic-boxes

        Args:
            matrix (torch.Tensor): (3, 3) general triclinc box matrix

        Returns:
            torch.Tensor: (3, 3) restricted triclinc box matrix
        """
        A = matrix[:, 0]
        B = matrix[:, 1]
        C = matrix[:, 2]
        ax = torch.linalg.norm(A)
        uA = A / ax
        bx = torch.dot(B, uA)
        by = torch.linalg.norm(torch.cross(uA, B))
        cx = torch.dot(C, uA)
        AxB = torch.cross(A, B)
        uAxB = AxB / torch.linalg.norm(AxB)
        cy = torch.dot(C, torch.cross(uAxB, uA))
        cz = torch.dot(C, uAxB)
        # validation code
        # import numpy.testing as npt
        # gamma = torch.arccos(torch.dot(A, C) / torch.linalg.norm(A) / torch.linalg.norm(C))
        # beta = torch.arccos(torch.dot(A, B) / torch.linalg.norm(A) / torch.linalg.norm(B))
        # npt.assert_allclose(
        #     bx,
        #     torch.linalg.norm(B) * torch.cos(gamma),
        #     err_msg=f"{bx} != {torch.linalg.norm(B) * torch.cos(gamma)}",
        # )
        # npt.assert_allclose(
        #     by,
        #     torch.linalg.norm(B) * torch.sin(gamma),
        #     err_msg=f"{by} != {torch.linalg.norm(B) * torch.sin(gamma)}",
        # )
        # npt.assert_allclose(
        #     cx,
        #     torch.linalg.norm(C) * torch.cos(beta),
        #     err_msg=f"{cx} != {torch.linalg.norm(C) * torch.cos(beta)}",
        # )
        # npt.assert_allclose(
        #     cy,
        #     (torch.dot(B, C) - bx * cx) / by,
        #     err_msg=f"{cy} != {(torch.dot(B, C) - bx * cx) / by}",
        # )
        # npt.assert_allclose(
        #     cz,
        #     torch.sqrt(torch.linalg.norm(C) ** 2 - cx**2 - cy**2),
        #     err_msg=f"{cz} != {torch.sqrt(torch.linalg.norm(C) ** 2 - cx ** 2 - cy ** 2)}",
        # )
        # TODO: extract origin and direction
        return torch.tensor([[ax, bx, cx], [0, by, cy], [0, 0, cz]])

    @staticmethod
    def calc_matrix_from_lengths_angles(
        lengths: torch.Tensor, angles: torch.Tensor
    ) -> torch.Tensor:
        """
        Get restricted triclinic box matrix from lengths and angles

        Args:
            lengths (torch.Tensor): lengths of box edges
            angles (torch.Tensor): angles between box edges in degree

        Returns:
            torch.Tensor: restricted triclinic box matrix
        """
        a, b, c = lengths
        alpha, beta, gamma = torch.deg2rad(angles)
        lx = a
        ly = b * torch.sin(gamma)
        xy = b * torch.cos(gamma)
        xz = c * torch.cos(beta)
        yz = (b * c * torch.cos(alpha) - xy * xz) / ly
        lz = torch.sqrt(c**2 - xz**2 - yz**2)
        return torch.tensor([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])

    @staticmethod
    def calc_matrix_from_size_tilts(sizes, tilts) -> torch.Tensor:
        """
        Get restricted triclinic box matrix from sizes and tilts

        Args:
            sizes (torch.Tensor): sizes of box edges
            tilts (torch.Tensor): tilts between box edges

        Returns:
            torch.Tensor: restricted triclinic box matrix
        """
        lx, ly, lz = sizes
        xy, xz, yz = tilts
        return torch.tensor([[lx, xy, xz], [0, ly, yz], [0, 0, lz]])

    @staticmethod
    def calc_lengths_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(matrix, axis=1)

    @staticmethod
    def calc_angles_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
        a = torch.linalg.norm(matrix[:, 0])
        b = torch.linalg.norm(matrix[:, 1])
        c = torch.linalg.norm(matrix[:, 2])
        alpha = torch.arccos((matrix[:, 1] @ matrix[:, 2]) / b / c)
        beta = torch.arccos((matrix[:, 0] @ matrix[:, 2]) / a / c)
        gamma = torch.arccos((matrix[:, 0] @ matrix[:, 1]) / a / b)
        return torch.rad2deg(torch.tensor([alpha, beta, gamma]))

    @staticmethod
    def calc_style_from_matrix(matrix: torch.Tensor) -> Style:

        if torch.allclose(matrix, torch.zeros((3, 3))):
            return Box.Style.FREE
        elif torch.allclose(matrix, torch.diag(torch.diagonal(matrix))):
            return Box.Style.ORTHOGONAL
        elif torch.tril(matrix, 1).sum() == 0:
            return Box.Style.TRICLINIC
        else:
            ValueError("Invalid box matrix")

    def set_lengths(self, lengths: torch.Tensor):
        self._matrix = self.calc_matrix_from_lengths_angles(lengths, self.angles)

    def set_angles(self, angles: torch.Tensor):
        self._matrix = self.calc_matrix_from_lengths_angles(self.lengths, angles)

    def set_matrix(self, matrix: torch.Tensor):
        self._matrix = matrix

    def set_lengths_angles(self, lengths: torch.Tensor, angles: torch.Tensor):
        self._matrix = self.calc_matrix_from_lengths_angles(lengths, angles)

    def set_lengths_tilts(self, lengths: torch.Tensor, tilts: torch.Tensor):
        self._matrix = self.calc_matrix_from_size_tilts(lengths, tilts)

    @property
    def volume(self) -> float:
        match self.style:
            case Box.Style.FREE:
                return 0
            case Box.Style.ORTHOGONAL:
                return torch.prod(self.lengths)
            case Box.Style.TRICLINIC:
                return torch.abs(torch.det(self._matrix))

    def get_distance_between_faces(self) -> torch.Tensor:
        match self.style:
            case Box.Style.FREE:
                return torch.zeros(3)
            case Box.Style.ORTHOGONAL:
                return self.lengths
            case Box.Style.TRICLINIC:
                a = self._matrix[:, 0]
                b = self._matrix[:, 1]
                c = self._matrix[:, 2]

                na = torch.cross(b, c)
                nb = torch.cross(c, a)
                nc = torch.cross(a, b)
                na /= torch.linalg.norm(na)
                nb /= torch.linalg.norm(nb)
                nc /= torch.linalg.norm(nc)

                return torch.tensor(
                    [torch.dot(na, a), torch.dot(nb, b), torch.dot(nc, c)]
                )

    def wrap(self, xyz: torch.Tensor) -> torch.Tensor:

        match self.style:
            case Box.Style.FREE:
                return self.wrap_free(xyz)
            case Box.Style.ORTHOGONAL:
                return self.wrap_orthogonal(xyz)
            case Box.Style.TRICLINIC:
                return self.wrap_triclinic(xyz)

    def wrap_free(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz

    def wrap_orthogonal(self, xyz: torch.Tensor) -> torch.Tensor:
        lengths = self.lengths
        return xyz - torch.floor(xyz / lengths) * lengths

    def wrap_triclinic(self, xyz: torch.Tensor) -> torch.Tensor:
        fractional = torch.dot(self.get_inv(), xyz.T)
        return torch.dot(self._matrix, fractional - torch.floor(fractional)).T

    def get_inv(self) -> torch.Tensor:
        return torch.linalg.inv(self._matrix)

    def diff_dr(self, dr: torch.Tensor) -> torch.Tensor:

        match self.style:
            case Box.Style.FREE:
                return dr
            case Box.Style.ORTHOGONAL | Box.Style.TRICLINIC:
                fractional = self.make_fractional(dr)
                fractional -= torch.round(fractional)
                return torch.dot(self._matrix, fractional.T).T

    def diff(self, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
        return self.diff_dr(r1 - r2)

    def make_fractional(self, r: torch.Tensor) -> torch.Tensor:
        return torch.dot(r, self.get_inv())

    def make_absolute(self, r: torch.Tensor) -> torch.Tensor:
        return torch.dot(r, self._matrix)

    def isin(self, xyz: torch.Tensor) -> bool:
        return torch.all(torch.abs(self.wrap(xyz) - xyz) < 1e-5)
