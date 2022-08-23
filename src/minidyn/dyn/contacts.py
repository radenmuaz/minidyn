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
from minidyn.dyn import functions as Fn
from minidyn.dyn.body import Shape

@register_pytree_node_class
class RigidContact:

    def __init__(self, body1_id=0, body2_id=0,shape1_id=0, shape2_id=0, colfunc_id=0):
        self.body1_id = body1_id
        self.body2_id = body2_id
        self.shape1_id = shape1_id
        self.shape2_id = shape2_id
        self.colfunc_id = colfunc_id

    def connect(self, world, body1, body2, shape1, shape2):
        self.body1_id = world.bodies.index(body1)
        self.body2_id = world.bodies.index(body2)
        self.shape1_id = body1.shapes.index(shape1)
        self.shape2_id = body2.shapes.index(shape2)

    @classmethod
    def solve_world(cls, world, qs, qds):
        
        rigid_contacts = world.rigid_contacts
        body1_ids = Fn.stack_attr(rigid_contacts, 'body1_id')
        body2_ids = Fn.stack_attr(rigid_contacts, 'body2_id')
        shape1_ids = Fn.stack_attr(rigid_contacts, 'shape1_id')
        shape2_ids = Fn.stack_attr(rigid_contacts, 'shape2_id')
        colfunc_ids = Fn.stack_attr(rigid_contacts, 'colfunc_id')

        q1s, q2s = qs[body1_ids], qs[body2_ids]
        qd1s, qd2s = qds[body1_ids], qds[body2_ids]

        # batchify shapes into two pytrees
        shapes = Fn.tree_stack(Fn.tree_stack(world.bodies).shapes)
        shape1_attrs, shape2_attrs = [], []
        for attr in shapes.tree_flatten()[0]:
            shape1_attrs += [attr[shape1_ids, body1_ids]]
            shape2_attrs += [attr[shape2_ids, body2_ids]]
        shape1s = Shape.tree_unflatten(None, shape1_attrs)
        shape2s = Shape.tree_unflatten(None, shape2_attrs)

        result = jax.vmap(cls.solve)(q1s, q2s, qd1s, qd2s, shape1s, shape2s, colfunc_ids)
        result['body1_id'] = body1_ids
        result['body2_id'] = body2_ids
        return result

    # @partial(jax.jit, static_argnums=(0,))
    @classmethod
    def solve(cls, q1, q2, qd1, qd2, shape1, shape2, colfunc_id):
        def get_contacts(q, qd, shape1, shape2, colfunc_id):
            q1, q2 = q[:7], q[7:]
            qd1, qd2 = qd[:7], qd[7:]
            col_dict = jax.lax.switch(colfunc_id, [separating_axis,], q1, q2, shape1, shape2)
            did_collide = col_dict['did_collide']
            p_ref = col_dict['p_ref']
            p_in = col_dict['p_in']
            n_ref = col_dict['n_ref']

            def get_depth(collide_flag, p_ref, p_in, vec):
                return jnp.where(collide_flag, jnp.dot(p_ref-p_in, vec), 0)
            
            C_pen = get_depth(did_collide, p_ref, p_in, n_ref)

            v_ref, v_in = qd1[4:], qd2[4:]
            v = jnp.where((v_ref==0).all(), -v_in, -v_ref) # skip to use incidence vel if 0

            def to_plane(v, n):
                return v - n*(jnp.dot(v, n) / jnp.linalg.norm(n))
            u_ref = to_plane(v, n_ref)
            C_fric = get_depth(did_collide, p_ref, p_in, u_ref)
            C = jnp.stack([C_pen, C_fric])
            return C, (C, col_dict)
        def get_Cd(f, q, qd, shape1, shape2, colfunc_id):
            J , (C, col_info) = jax.jacfwd(f, 0, True)(q, qd, shape1, shape2, colfunc_id)
            Cd = J@qd
            return Cd, (J, Cd, C, col_info)
        
        q = jnp.concatenate([q1, q2])
        qd = jnp.concatenate([qd1, qd2])
        Jd, (J, Cd, C, col_info) = \
             jax.jacfwd(partial(get_Cd, get_contacts), 0, True)(q, qd, shape1, shape2, colfunc_id)
        Jd_pen, Jd_fric = jnp.split(Jd, 2, 0)
        J_pen, J_fric = jnp.split(J, 2, 0)
        Cd_pen, Cd_fric = jnp.split(Cd, 2, 0)
        C_pen, C_fric = jnp.split(C, 2, 0)
        return dict(Jd_pen=Jd_pen, Jd_fric=Jd_fric,
                    J_pen=J_pen, J_fric=J_fric,
                    Cd_pen=Cd_pen, Cd_fric=Cd_fric,
                    C_pen=C_pen, C_fric=C_fric,
                    col_info=col_info)
     
    def tree_flatten(self):
        children = (
            self.body1_id,
            self.body2_id,
            self.shape1_id,
            self.shape2_id,
            self.colfunc_id
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
