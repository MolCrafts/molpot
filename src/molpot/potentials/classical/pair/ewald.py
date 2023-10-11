import torch
from molpot import kw

class EnergyEwald(torch.nn.Module):
    """
    Compute the Coulomb energy of a set of point charges inside a periodic box.
    Only works for periodic boundary conditions in all three spatial directions and orthorhombic boxes.

    Args:
        alpha (float): Ewald alpha.
        k_max (int): Number of lattice vectors.
        energy_unit (str/float): Units used for the energy.
        position_unit (str/float): Units used for lengths and positions.
        output_key (str): Name of the energy property in the output.
        charges_key (str): Key of partial charges in the input batch.
        use_long_range_nblist (bool): Whether to use standard or long range neighbor list elements (default = True).
        screening_fn (optional, float): Apply a screening function to the real space interaction.
    """

    def __init__(
        self,
        alpha: float,
        k_max: int,
        output_key: str,
        use_long_range_nblist: bool = True,
        screening_fn: Optional[nn.Module] = None,
    ):
        super(EnergyEwald, self).__init__()

        # Get the appropriate Coulomb constant
        # ke is convert unit for Ha * Bohr
        # ke = spk_units.convert_units("Ha", energy_unit) * spk_units.convert_units(
        #     "Bohr", position_unit
        # )
        # self.register_buffer("ke", torch.Tensor([ke]))

        self.charges_key = charges_key
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.use_long_range_nblist = use_long_range_nblist

        self.screening_fn = screening_fn

        # TODO: automatic computation of alpha
        self.register_buffer("alpha", torch.Tensor([alpha]))

        # Set up lattice vectors
        self.k_max = k_max
        kvecs = self._generate_kvecs()
        self.register_buffer("kvecs", kvecs)

    def _generate_kvecs(self) -> torch.Tensor:
        """
        Auxiliary routine for setting up the k-vectors.

        Returns:
            torch.Tensor: k-vectors.
        """
        krange = torch.arange(0, self.k_max + 1, dtype=self.alpha.dtype)
        krange = torch.cat([krange, -krange[1:]])
        kvecs = torch.cartesian_prod(krange, krange, krange)
        norm = torch.sum(kvecs**2, dim=1)
        kvecs = kvecs[norm <= self.k_max**2 + 2, :]
        norm = norm[norm <= self.k_max**2 + 2]
        kvecs = kvecs[norm != 0, :]

        return kvecs

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute the Coulomb energy of the periodic system.

        Args:
            inputs (dict(str,torch.Tensor)): Input batch.

        Returns:
            dict(str, torch.Tensor): results with Coulomb energy.
        """
        q = inputs[kw.charge].squeeze(-1)
        idx_m = inputs[kw.idx_m]

        # Use long range neighbor list if requested
        if self.use_long_range_nblist:
            r_ij = inputs[kw.Rij_lr]
            idx_i = inputs[kw.idx_i_lr]
            idx_j = inputs[kw.idx_j_lr]
        else:
            r_ij = inputs[kw.Rij]
            idx_i = inputs[kw.idx_i]
            idx_j = inputs[kw.idx_j]

        d_ij = torch.norm(r_ij, dim=1)

        positions = inputs[kw.R]
        box = inputs[kw.box]

        n_atoms = q.shape[0]
        n_molecules = int(idx_m[-1]) + 1

        # Get real space and reciprocal space contributions
        y_real = self._real_space(q, d_ij, idx_i, idx_j, idx_m, n_atoms, n_molecules)
        y_reciprocal = self._reciprocal_space(q, positions, box, idx_m, n_molecules)

        y = y_real + y_reciprocal

        inputs[self.output_key] = y
        return inputs

    def _real_space(
        self,
        q: torch.Tensor,
        d_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        idx_m: torch.Tensor,
        n_atoms: int,
        n_molecules: int,
    ) -> torch.Tensor:
        """
        Compute the real space contribution of the screened charges.

        Args:
            q (torch.Tensor): Partial charges.
            d_ij (torch.Tensor): Interatomic distances.
            idx_i (torch.Tensor): Indices of atoms i in the distance pairs.
            idx_j (torch.Tensor): Indices of atoms j in the distance pairs.
            idx_m (torch.Tensor): Molecular indices of each atom.
            n_atoms (int): Total number of atoms.
            n_molecules (int): Number of molecules.

        Returns:
            torch.Tensor: Real space Coulomb energy.
        """

        # Apply erfc for Ewald summation
        f_erfc = torch.erfc(torch.sqrt(self.alpha) * d_ij)
        # Combine functions and multiply with inverse distance
        f_r = f_erfc / d_ij

        # Apply screening function
        if self.screening_fn is not None:
            screen = self.screening_fn(d_ij)
            f_r = f_r * (1.0 - screen)

        potential_ij = q[idx_i] * q[idx_j] * f_r

        y = snn.scatter_add(potential_ij, idx_i, dim_size=n_atoms)
        y = snn.scatter_add(y, idx_m, dim_size=n_molecules)
        y = 0.5 * self.ke * y.squeeze(-1)

        return y

    def _reciprocal_space(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        box: torch.Tensor,
        idx_m: torch.Tensor,
        n_molecules: int,
    ):
        """
        Compute the reciprocal space contribution.

        Args:
            q (torch.Tensor): Partial charges.
            positions (torch.Tensor): Atom positions.
            box (torch.Tensor): Molecular boxs.
            idx_m (torch.Tensor): Molecular indices of each atom.
            n_molecules (int): Number of molecules.

        Returns:
            torch.Tensor: Real space Coulomb energy.
        """
        # extract box dimensions from boxs
        recip_box = 2.0 * np.pi * torch.linalg.inv(box).transpose(1, 2)
        v_box = torch.abs(torch.linalg.det(box))

        if torch.any(torch.isclose(v_box, torch.zeros_like(v_box))):
            raise EnergyEwaldError("Simulation box has no volume.")

        # 1) compute the prefactor
        prefactor = 2.0 * np.pi / v_box

        # setup kvecs M x K x 3
        kvecs = torch.matmul(self.kvecs[None, :, :], recip_box)

        # Squared length of vectors M x K
        k_squared = torch.sum(kvecs**2, dim=2)

        # 2) Gaussian part of ewald sum
        q_gauss = torch.exp(-0.25 * k_squared / self.alpha)  # M x K

        # 3) Compute charge density fourier terms
        # Dot product in exponent -> MN x K, expand kvecs in MN batch structure
        kvec_dot_pos = torch.sum(kvecs[idx_m] * positions[:, None, :], dim=2)

        # charge densities MN x K -> M x K
        q_real = snn.scatter_add(
            (q[:, None] * torch.cos(kvec_dot_pos)), idx_m, dim_size=n_molecules
        )
        q_imag = snn.scatter_add(
            (q[:, None] * torch.sin(kvec_dot_pos)), idx_m, dim_size=n_molecules
        )
        # Compute square of density
        q_dens = q_real**2 + q_imag**2

        # Sum over k vectors -> M x K -> M
        y_ewald = prefactor * torch.sum(q_dens * q_gauss / k_squared, dim=1)

        # 4) self interaction correction -> MN
        self_interaction = torch.sqrt(self.alpha / np.pi) * snn.scatter_add(
            q**2, idx_m, dim_size=n_molecules
        )

        # Bring everything together
        y_ewald = self.ke * (y_ewald - self_interaction)

        return y_ewald