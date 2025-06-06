from molpot import Frame, alias, get_logger
from ..event import MDEvents
from ignite.engine import Engine
from ..handler import MDHandler

logger = get_logger("molpot.md")

class Integrator(MDHandler):
    """
    Basic integrator class template. Uses the typical scheme of propagating
    frame momenta in two half steps and frame positions in one main step.
    The half steps are defined by default and only the _main_step function
    needs to be specified. Uses atomic time units internally.

    If required, the torch graphs generated by this routine can be detached
    every step via the detach flag.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    def __init__(self, name, events, priorities, time_step: float, ):
        super().__init__(name, events, priorities)
        # Convert fs to internal time units.
        self.time_step = time_step


class VelocityVerlet(Integrator):
    """
    Standard velocity Verlet integrator for non ring-polymer simulations.

    Args:
        time_step (float): Integration time step in femto seconds.
    """

    def __init__(self, time_step: float, name: str = "velocity_verlet"):
        super().__init__(
            name,
            {
                MDEvents.STARTED,
                MDEvents.INITIAL_INTEGRATE,
                MDEvents.POST_INTEGRATE,
                MDEvents.FINAL_INTEGRATE,
            },
            (0, 0, 0, 0),
            time_step,
        )

    def on_started(self, engine):
        engine.integrator = self

    def on_initial_integrate(self, engine: Engine):
        r"""
        Half steps propagating the frame momenta according to:

        ..math::
            p = p + \frac{1}{2} F \delta t

        Args:
            frame (schnetpack.md.Frame): Frame class containing all molecules and their
                             replicas.
        """
        frame = engine.state.frame
        frame["atoms", "momenta"] = (
            frame["atoms", "momenta"] + 0.5 * frame["predicts", "forces"] * self.time_step
        )

    def on_post_integrate(self, engine: Engine):
        r"""
        Propagate the positions of the frame according to:

        ..math::
            q = q + \frac{p}{m} \delta t

        Args:
            frame (schnetpack.md.Frame): Frame class containing all molecules and their
                             replicas.
        """
        frame = engine.state.frame
        frame[alias.R] = (
            frame[alias.R]
            + self.time_step * frame["atoms", "momenta"] / frame["atoms", "mass"]
        )

    on_final_integrate = on_initial_integrate
