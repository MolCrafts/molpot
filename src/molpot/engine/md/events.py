from ignite.engine import Engine, EventEnum


class MDMainEvents(EventEnum):
    """
    timestep execution events. The protocol is adapted from: https://docs.lammps.org/Developer_flow.html
    """

    PRE_FORCE = "pre_force"
    FORCE = "force"
    POST_FORCE = "post_force"
    POST_INTEGRATE = "post_integrate"
    FINAL_INTEGRATE = "final_integrate"
    END_STEP = "end_step"
    INITIAL_INTEGRATE = "initial_integrate"
    PRE_NEIGHBOR = "pre_neighbor"
    NEIGHBOR = "neighbor"
    POST_NEIGHBOR = "post_neighbor"