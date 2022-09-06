import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
from minidyn.dyn.body import Body
from typing import *

from minidyn.dyn import functions as Fn

def euler(q, v, vd, dt=1e-3):
    v_new = v + dt*vd
    qd = Fn.qv2qd(q, v)
    q_new = q + dt*qd
    q_new = Fn.quat_norm(q_new)

    return q_new, v_new