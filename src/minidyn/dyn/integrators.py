import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
from minidyn.dyn.body import Body
from typing import *

from jax import vmap, tree_multimap,lax

def euler(q, qd, qdd, dt=1e-3):
    qd_new = qd + dt*qdd
    q_new = q + dt*qd_new

    return q_new, qd_new

    dx = qd[:, 4:]
    quat = q[:, :4]
    quatd = qd[:, :4]
    dquat  = (0.5*quat*quatd)
    dq = jnp.concatenate((dquat,dx),1)
    
    q_new = q + dt*dq
    qd_new = qd + dt*qdd

    return q_new, qd_new