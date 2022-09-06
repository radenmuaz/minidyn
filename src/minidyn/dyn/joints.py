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
# from minidyn.dyn import WorldC
import minidyn.dyn.functions as Fn

@register_pytree_node_class
class FixedJoint:
    def __init__(self, body1_id=0, body2_id=0,
      anchor1=jnp.array((0.0,0.0,0.0)),
      anchor2=jnp.array((0.0,0.0,0.0))):
        self.body1_id = body1_id
        self.body2_id = body2_id
        self.anchor1 = anchor1
        self.anchor2 = anchor2

    def connect(self, world, body1, body2, anchor1, anchor2):
        self.body1_id = world.bodies.index(body1)
        self.body2_id = world.bodies.index(body2)
        self.anchor1 = anchor1
        self.anchor2 = anchor2
    
    @classmethod
    def solve_world(cls, world, qs, vs):
        return partial(cls._solve_world, return_d=False)(world, qs, vs)  
    
    @classmethod
    def solve_world_d(cls, world, qs, vs):
        return partial(cls._solve_world, return_d=True)(world, qs, vs)

    @classmethod
    def _solve_world(cls, world, qs, vs, return_d=True):
          
        fixed_joints = world.fixed_joints
        body1_ids = Fn.stack_attr(fixed_joints, 'body1_id')
        body2_ids = Fn.stack_attr(fixed_joints, 'body2_id')
        anchor1s = Fn.stack_attr(fixed_joints, 'anchor1')
        anchor2s = Fn.stack_attr(fixed_joints, 'anchor2')

        q1s, q2s = qs[body1_ids], qs[body2_ids]
        v1s, v2s = vs[body1_ids], vs[body2_ids]

        # result = jax.vmap(cls.solve)(q1s, q2s, qd1s, qd2s, shape1s, shape2s, colfunc_ids)
        if return_d:
            result = jax.vmap(cls.solve_with_d)(q1s, q2s, v1s, v2s, anchor1s, anchor2s)
        else:
            result = jax.vmap(cls.solve_without_d)(q1s, q2s, v1s, v2s, anchor1s, anchor2s)
        result['body1_id'] = body1_ids
        result['body2_id'] = body2_ids

        return result
    
    @classmethod
    def get_constraints(cls, q, anchor1, anchor2):
        q1, q2 = q[:7], q[7:]

        def get_trans(q1, q2, anchor1, anchor2):
            x1, x2 = q1[4:], q2[4:]
            ret = (x1 + anchor1) - (x2 + anchor2)
            return ret
        def get_rot(q1, q2):
            quat1, quat2 = q1[:4], q2[:4]
            quat1, quat2 = Fn.quat_norm(quat1), Fn.quat_norm(quat2)

            # ret = (quat1 * Fn.quat_inv(quat2)) - jnp.array([1, 0, 0, 0])
            ret = jnp.array([1, 0, 0, 0]) - (quat1 * Fn.quat_inv(quat2))
            # ret = quat1 * Fn.quat_inv(quat2)
            return ret

        C_trans = get_trans(q1, q2, anchor1, anchor2)
        C_rot = get_rot(q1, q2)
        C = jnp.concatenate([C_rot,C_trans])
        return C, (C, )
    
    @classmethod
    def solve_with_d(cls, q1, q2, v1, v2, anchor1, anchor2):
        def get_J_col(J_q_col, q_col):
            Jang1 = J_q_col[0:4]@Fn.get_jac_quatdot(q_col[0:4])
            Jlin1 = J_q_col[4:7]
            Jang2 = J_q_col[7:11]@Fn.get_jac_quatdot(q_col[7:11])
            Jlin2 = J_q_col[11:14]
            return jnp.concatenate([Jang1, Jlin1, Jang2, Jlin2])
        def get_Cd(f, q, v, anchor1, anchor2):
            J_q, (C,) = jax.jacfwd(f, 0, True)(q, anchor1, anchor2)
            
            J = jax.vmap(get_J_col)(J_q, jnp.repeat(q[jnp.newaxis,:], len(J_q), 0))
            Cd = J@v
            # breakpoint()
            return Cd, (J, Cd, C)
        q = jnp.concatenate([q1, q2])
        v = jnp.concatenate([v1, v2])
        Jd_q, (J, Cd, C) = \
             jax.jacfwd(partial(get_Cd, cls.get_constraints), 0, True)(q, v, anchor1, anchor2)
        Jd = jax.vmap(get_J_col)(Jd_q, jnp.repeat(q[jnp.newaxis,:], len(Jd_q), 0))
        dCddv = jax.jacfwd(partial(get_Cd, cls.get_constraints), 1, True)(q, v, anchor1, anchor2)[0]
        return dict(Jd=Jd, J=J, Cd=Cd, C=C, dCddv=dCddv)
    
    def tree_flatten(self):
        children = (
                    self.body1_id,
                    self.body2_id,
      self.anchor1,
      self.anchor2,
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