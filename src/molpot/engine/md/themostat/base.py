import torch
from ..event import MDEvents
from ..handler import MDHandler
from .utils import YSWeights
import molpot as mpot
from torch.nn.parameter import UninitializedParameter
from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin

config = mpot.get_config()

kB = 1


class Thermostat(MDHandler, Module, LazyModuleMixin):

    def __init__(self, name, temperature: float, time_constant: float):
        super().__init__(
            name, {MDEvents.STARTED, MDEvents.START_STEP, MDEvents.END_STEP}, (1, 1, 1)
        )
        Module.__init__(self)
        LazyModuleMixin.__init__(self)
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("time_constant", torch.tensor(time_constant))

    def on_start_step(self, engine):
        self._apply_thermostat(engine.state.frame)

    def on_end_step(self, engine):
        self._apply_thermostat(engine.state.frame)

    def _apply_thermostat(self, frame):
        raise NotImplementedError


class NoseHoverThermostat(Thermostat):

    def __init__(
        self,
        temperature: float,
        time_constant: float,
        chain_length: int = 3,
        massive: bool = False,
        multi_step: int = 2,
        integration_order: int = 3,
    ):
        super().__init__(temperature=temperature, time_constant=time_constant)

        self.register_buffer("chain_length", torch.tensor(chain_length))
        self.register_buffer("massive", torch.tensor(massive))
        self.register_buffer("frequency", torch.tensor(1.0 / time_constant))

        self.register_buffer("kb_temperature", self.temperature * kB)
        self.register_buffer("multi_step", torch.tensor(multi_step))
        self.register_buffer("integration_order", torch.tensor(integration_order))
        self.register_buffer("time_step", UninitializedParameter())

        # Find out number of particles (depends on whether massive or not)
        self.register_buffer("degrees_of_freedom", UninitializedParameter())
        self.register_buffer("masses", UninitializedParameter())

        self.register_buffer("velocities", UninitializedParameter())
        self.register_buffer("positions", UninitializedParameter())
        self.register_buffer("forces", UninitializedParameter())

    def on_started(self, engine):
        frame = engine.frame
        integration_weights = YSWeights().get_weights(self.integration_order.item())

        self.time_step = (
            engine.integrator.time_step * integration_weights / self.multi_step
        )

        # Determine shape of tensors and internal degrees of freedom
        n_atoms_total, n_atoms = frame.n_atoms
        n_molecules = frame.n_molecules

        if self.massive:
            state_dimension = (n_atoms_total, 3, self.chain_length)
            self.degrees_of_freedom = torch.ones(
                (n_atoms_total, 3),
                device=config.device,
                dtype=config.dtype,
            )
        else:
            state_dimension = (n_molecules, 1, self.chain_length)
            self.degrees_of_freedom = (3 * n_atoms[:, None]).to(
                device=config.device,
                dtype=config.dtype,
            )

        # Set up masses
        self._init_masses(state_dimension)

        # Set up internal variables
        self.positions = torch.zeros(
            state_dimension, device=config.device, dtype=config.dtype
        )
        self.forces = torch.zeros(
            state_dimension, device=config.device, dtype=config.dtype
        )
        self.velocities = torch.zeros(
            state_dimension, device=config.device, dtype=config.dtype
        )

    def _init_masses(self, state_dimension):

        self.masses = torch.ones(
            state_dimension, device=config.device, dtype=config.dtype
        )

        # Get masses of innermost thermostat
        self.masses[..., 0] = (
            self.degrees_of_freedom * self.kb_temperature / self.frequency**2
        )
        # Set masses of remaining thermostats
        self.masses[..., 1:] = self.kb_temperature / self.frequency**2

    def _propagate_thermostat(self, kinetic_energy: torch.tensor) -> torch.tensor:

        # Compute forces on first thermostat
        self.forces[..., 0] = (
            kinetic_energy - self.degrees_of_freedom * self.kb_temperature
        ) / self.masses[..., 0]

        scaling_factor = 1.0

        for _ in range(self.multi_step):
            for idx_ys in range(self.integration_order):
                time_step = self.time_step[idx_ys]

                # Update velocities of outermost bath
                self.velocities[..., -1] += 0.25 * self.forces[..., -1] * time_step

                # Update the velocities moving through the beads of the chain
                for chain in range(self.chain_length - 2, -1, -1):
                    coeff = torch.exp(
                        -0.125 * time_step * self.velocities[..., chain + 1]
                    )
                    self.velocities[..., chain] = (
                        self.velocities[..., chain] * coeff**2
                        + 0.25 * self.forces[..., chain] * coeff * time_step
                    )

                # Accumulate velocity scaling
                scaling_factor *= torch.exp(-0.5 * time_step * self.velocities[..., 0])
                # Update forces of innermost thermostat
                self.forces[..., 0] = (
                    scaling_factor * scaling_factor * kinetic_energy
                    - self.degrees_of_freedom * self.kb_temperature
                ) / self.masses[..., 0]

                # Update thermostat positions
                # Only required if one is interested in the conserved
                # quantity of the NHC.
                # self.positions += 0.5 * self.velocities * time_step

                # Update the thermostat velocities
                for chain in range(self.chain_length - 1):
                    coeff = torch.exp(
                        -0.125 * time_step * self.velocities[..., chain + 1]
                    )
                    self.velocities[..., chain] = (
                        self.velocities[..., chain] * coeff**2
                        + 0.25 * self.forces[..., chain] * coeff * time_step
                    )
                    self.forces[..., chain + 1] = (
                        self.masses[..., chain] * self.velocities[..., chain] ** 2
                        - self.kb_temperature
                    ) / self.masses[..., chain + 1]

                # Update velocities of outermost thermostat
                self.velocities[..., -1] += 0.25 * self.forces[..., -1] * time_step

        return scaling_factor

    def _compute_kinetic_energy(self, frame):
        """
        Routine for computing the kinetic energy of the innermost NH thermostats based on the momenta and masses of the
        simulated frames.

        Args:
            frame (molpot.System): System object.

        Returns:
            torch.Tensor: Kinetic energy associated with the innermost NH thermostats. These are summed over the
                          corresponding degrees of freedom, depending on whether a massive NHC is used.

        """
        if self.massive:
            # Compute the kinetic energy (factor of 1/2 can be removed, as it
            # cancels with a times 2)
            kinetic_energy = frame.momenta**2 / frame.masses
            return kinetic_energy
        else:
            return 2.0 * frame.kinetic_energy

    def _apply_thermostat(self, engine):
        """
        Propagate the NHC thermostat, compute the corresponding scaling factor and apply it to the momenta of the
        system. If a normal mode transformer is provided, this is done in the normal model representation of the ring
        polymer.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Get kinetic energy (either for massive or normal degrees of freedom)
        kinetic_energy = self._compute_kinetic_energy(engine.frame)

        # Accumulate scaling factor
        scaling_factor = self._propagate_thermostat(kinetic_energy)

        # Update system momenta
        if not self.massive:
            scaling_factor = scaling_factor[..., None, ...]

        engine.frame.momenta = engine.frame.momenta * scaling_factor
