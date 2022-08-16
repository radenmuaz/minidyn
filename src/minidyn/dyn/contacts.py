from typing import *
from functools import partial

import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# from jax import vmap, tree_map,lax
from jax.tree_util import tree_map
from jax.tree_util import register_pytree_node_class
from pyparsing import col

from minidyn.dyn.collisions import separating_axis

@register_pytree_node_class
class RigidContact:

    def __init__(self, ib_pair=[], shape_pair=[]):
        self.ib_pair = ib_pair
        self.shape_pair = shape_pair

    def connect(self, world, body1, body2, shape1, shape2):
        ib1, ib2 = world.bodies.index(body1), world.bodies.index(body2)
        self.ib_pair = [ib1, ib2]
        self.shape_pair = [shape1, shape2]


    # @partial(jax.jit, static_argnums=(0,))
    def __call__(self, q, qd, solve_func):
        

        def get_contacts(q, qd):
            col_dict = solve_func(q, self.ib_pair, self.shape_pair)
            did_collide = col_dict['did_collide']
            p_ref = col_dict['p_ref']
            p_in = col_dict['p_in']
            n_ref = col_dict['n_ref']

            def get_depth(collide_flag, p_ref, p_in, vec):
                return jnp.where(collide_flag, jnp.dot(p_ref-p_in, vec), 0)
            
            C_pen = get_depth(did_collide, p_ref, p_in, n_ref)

            ib1, ib2 = self.ib_pair
            v_ref, v_in = qd[ib1,4:], qd[ib2,4:]
            v = jnp.where((v_ref==0).all(), -v_in, -v_ref) # skip to use incidence vel if 0

            def to_plane(v, n):
                return v - n*(jnp.dot(v, n) / jnp.linalg.norm(n))
            u_ref = to_plane(v, n_ref)
            C_fric = get_depth(did_collide, p_ref, p_in, u_ref)
            C = jnp.stack([C_pen, C_fric])
            return C, (C, col_dict)
        def get_J_Jd(f, q, qd):
            J, (C, col_info) = jax.jacfwd(f, 0, True)(q, qd)
            k = C.size
            N = q.size
            # Cd = J.reshape(k,N)@qd.reshape(N,1)
            Cd = (J*qd[jnp.newaxis,::]).sum(2).sum(1)
            return Cd, (J, Cd, C, col_info)
        
        Jd, (J, Cd, C, col_info) = jax.jacfwd(partial(get_J_Jd, get_contacts), 0, True)(q, qd)
        return Jd, J, Cd, C, col_info

     
    def tree_flatten(self):
        children = (
            self.ib_pair,
            self.shape_pair
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
