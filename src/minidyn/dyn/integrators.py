import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
from minidyn.dyn.body import Body
from typing import *


def euler(q, qd, qdd, dt=1e-3):
    qd_new = qd + dt*qdd
    q_new = q + dt*qd_new

    return q_new, qd_new