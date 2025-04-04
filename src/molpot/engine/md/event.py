from ignite.engine import EventEnum, Events


class MDEvents(EventEnum):
    """
    timestep execution events. The protocol is adapted from: https://docs.lammps.org/Developer_flow.html
    """
    STARTED = Events.STARTED.value

    START_STEP = "start_step"
    INITIAL_INTEGRATE = "initial_integrate"
    POST_INTEGRATE = "post_integrate"
    PRE_NEIGHBOR = "pre_neighbor"
    NEIGHBOR = "neighbor"
    POST_NEIGHBOR = "post_neighbor"
    PRE_FORCE = "pre_force"
    FORCE = "force"
    POST_FORCE = "post_force"
    FINAL_INTEGRATE = "final_integrate"
    END_STEP = "end_step"

    COMPLETED = Events.COMPLETED.value

