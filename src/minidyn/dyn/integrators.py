import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
from minidyn.dyn.body import Body
from typing import *

from jax import vmap, tree_multimap,lax

def euler(x, xd, dt=1e-3):
    return x + dt*xd