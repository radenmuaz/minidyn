from typing import *
from functools import partial

import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
from jax import vmap, tree_map,lax
from jax.tree_util import register_pytree_node_class
from pyparsing import col

from minidyn.dyn import integrators
from minidyn.dyn import collisions
from minidyn.dyn import contacts
from minidyn.dyn import World
import minidyn.dyn.functions as Fn

@register_pytree_node_class
class FixedJoint:
    def __init__(self, ib_pair=[], a_pair=[]):
        self.ib_pair = ib_pair
        self.a_pair = a_pair

    def connect(self, world, body1, body2, a1, a2):
        ib1, ib2 = world.bodies.index(body1), world.bodies.index(body2)
        self.ib_pair = [ib1, ib2]
        self.a_pair = [a1, a2]

    def __call__(self, q, qd):
        def collate(q):
            (ib1, ib2) = self.ib_pair
            q1 = q[ib1]#[jnp.newaxis,:]
            q2 = q[ib2]#[jnp.newaxis,:]
            return q1, q2
        def get_trans(q):
            (q1, q2), (a1, a2) = collate(q), self.a_pair
            x1, x2 = q1[4:], q2[4:]
            ret = (x1 + a1) - (x2 + a2)
            return ret, ret
        def get_rot(q):
            (q1, q2) = collate(q)
            quat1, quat2 = q1[:4], q2[:4]

            ret = (quat1 * Fn.quat_inv(quat2)) - jnp.array([1, 0, 0, 0])
            return ret, ret
        
        def get_J_Jd(f, q, qd):
            J, C = jax.jacfwd(f, 0, True)(q)
            k = C.size
            N = q.size
            # Cd = J.reshape(k,N)@qd.reshape(N,1)
            Cd = (J*qd[jnp.newaxis,::]).sum(2).sum(1)
            # breakpoint()
            return Cd, (J, Cd, C)

        # J_rot, C_rot = jax.jacfwd(get_rot, 0 , True)(q)
        # J_trans, C_trans = jax.jacfwd(get_trans, 0, True)(q)
        Jd_rot, (J_rot, Cd_rot, C_rot) = jax.jacfwd(partial(get_J_Jd, get_rot), 0, True)(q, qd)
        Jd_trans, (J_trans, Cd_trans, C_trans) = jax.jacfwd(partial(get_J_Jd, get_trans), 0, True)(q, qd)
        # return Jd_trans, J_trans, Cd_trans, C_trans
        Jd = jnp.concatenate([Jd_rot, Jd_trans])
        J = jnp.concatenate([J_rot, J_trans])
        Cd = jnp.concatenate([Cd_rot, Cd_trans])
        C = jnp.concatenate([C_rot, C_trans])
        return Jd, J, Cd, C 
    
    def tree_flatten(self):
        children = (
                    self.ib_pair,
                    self.a_pair
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    




        
'''
@register_pytree_node_class
class FixedJoint:
    def __init__(self, world, b1, b2, a1, a2):
        self.ib1 = world.bodies.index(b1)
        self.ib2 = world.bodies.index(b2)
        self.a1 = a1
        self.a2 = a2

        pass
    def __call__(self, q, Minv):
        q1 = q[self.ib1]
        q2 = q[self.ib2]
        a1 = self.a1
        a2 = self.a2
        def C(q1, q2, a1, a2):
            x1, x2 = q1[4:], q2[4:]
            trans = (x1 + a1) - (x2 + a2)
        
            quat1, quat2 = q1[:4], q2[:4]
            rot = (quat1 * Fn.quat_inv(quat2)) - jnp.array([1, 0, 0, 0])
            ret = jnp.concatenate([trans, rot])
            return ret, ret
        

        J_trans, C_trans = jax.jacfwd(partial(C_trans, q), argnums=0, has_aux=True)(q)
        J_rot, C_rot = jax.jacfwd(partial(C_rot, q), argnums=0, has_aux=True)(q)
        Minv = Minv[]
        K = J @ Minv @ J.T
        Kinv = jnp.linalg.inv(K)
        Lmult = Kinv @ J @ (Minv @ (u_vec - C @ qd_vec - g) + Jd@qd_vec)
        F_c = J.T @ -Lmult
        F_c = jnp.where(jnp.isnan(F_c), 0, F_c)
        return F_c
'''


'''
        def C(q1, q2, a1, a2):
            x1, x2 = q1[4:], q2[4:]
            trans = (x1 + a1) - (x2 + a2)
        
            quat1, quat2 = q1[:4], q2[:4]
            rot = (quat1 * Fn.quat_inv(quat2)) - jnp.array([1, 0, 0, 0])
            ret = jnp.concatenate([trans, rot])
            return ret, ret

        def C_trans(q1, q2, a1, a2):
            x1, x2 = q1[4:], q2[4:]
            ret = (x1 + a1) - (x2 + a2)
            return ret, ret
        
        def C_rot(q1, q2):
            quat1, quat2 = q1[:4], q2[:4]
            ret = (quat1 * Fn.quat_inv(quat2)) - jnp.array([1, 0, 0, 0])
            return ret, ret
'''