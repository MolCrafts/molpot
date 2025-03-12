from .base import Thermostat
import torch
from molpot import get_config, get_unit, alias
from torch.nn.parameter import UninitializedParameter

config = get_config()
unit = get_unit()


class Langevin(Thermostat):

    """
    Basic stochastic Langevin thermostat, see e.g. [#langevin_thermostat1]_ for more details.

    Args:
        temperature_bath (float): Temperature of the external heat bath in Kelvin.
        time_constant (float): Thermostat time constant in fs

    References
    ----------
    .. [#langevin_thermostat1] Bussi, Parrinello:
       Accurate sampling using Langevin dynamics.
       Physical Review E, 75(5), 056707. 2007.
    """

    def __init__(self, temperature: float, time_constant: float):
        
        super().__init__(temperature=temperature, time_constant=time_constant)

        self.register_buffer("thermostat_factor", UninitializedParameter())
        self.register_buffer("c1", UninitializedParameter())
        self.register_buffer("c2", UninitializedParameter())

    def on_started(self, engine):
        """
        Initialize the Langevin coefficient matrices based on the system and simulator properties.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        # Initialize friction coefficients
        gamma = (
            torch.ones(1, device=config.device, dtype=config.ftype)
            / self.time_constant
        )

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * engine.integrator.time_step * gamma)
        c2 = torch.sqrt(1 - c1**2)

        self.c1 = c1[:, None, None]
        self.c2 = c2[:, None, None]

        # Get mass and temperature factors
        self.thermostat_factor = torch.sqrt(
            engine.frame[alias.atom_mass] * unit.kB * self.temperature
        )

    def _apply_thermostat(self, frame):
        
        momenta = frame["atoms", "momenta"]
        thermostat_noise = torch.randn_like(momenta)
        frame["atoms", "momenta"] = (
            self.c1 * momenta + self.c2 * self.thermostat_factor * thermostat_noise
        )