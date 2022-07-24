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

class CompliantContacts:
    def __init__(self):
        pass
    def __call__(self, world, collision_solver, q, qd):
        collide_flags, mtvs, n_refs, p_refs, p_ins = collision_solver(world, q)

        collide_flags = collide_flags[world.body_pairs_idxs,:].squeeze()
        mtvs = mtvs[world.body_pairs_idxs,:].squeeze()
        n_refs = n_refs[world.body_pairs_idxs,:].squeeze()
        # import pdb;pdb.set_trace()

        evens = (jnp.arange(len(n_refs)) % 2 == 0)[:,jnp.newaxis]
        n_refs = jnp.where(evens, -n_refs, n_refs)
        p_refs = p_refs[world.body_pairs_idxs,:].squeeze()
        p_ins = p_ins[world.body_pairs_idxs,:].squeeze()
        # v = qd[world.body_pairs_idxs,4:]
        
        # def to_plane(v, n):
        #     return v - n*(jnp.dot(v, n) / jnp.linalg.norm(n))
        # u = jax.vmap(to_plane)(v, n_refs)
        # u = jnp.where(jnp.isnan(u), 0, u)

        ib, s = [], []
        for (ib1, ib2), (bs1, bs2) in zip(world.body_pairs_mat_idxs, world.shape_pairs_mat):
            ib += [ib1, ib2]
            s += [bs1, bs2]
        
        def stack_attr(pytrees, attr):
            return  tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[getattr(t, attr) for t in pytrees])

        Kp = stack_attr(s, 'Kp')
        ib = jnp.array(ib)

        def get_F(collide_flag, p_ref, p_in, dir, Kp):
            depth = jnp.dot(p_ref-p_in, dir)
            F = depth*dir*Kp
            # F = jnp.array([0, 0, 0 ,0, *F])
            return jnp.where(collide_flag, F, 0)
        def arrange_F(carry_F, F_and_ib):
            F, ib = F_and_ib
            Fe = jnp.array([0, 0, 0 ,0, *F])
            return carry_F.at[ib].add(Fe), Fe
        F = jax.vmap(get_F)(collide_flags, p_refs, p_ins, n_refs, Kp)
        F, _ = jax.lax.scan(arrange_F, jnp.zeros(q.shape), (F, ib))


        

        return F

        
class RigidContacts:
    def __init__(self):
        pass
    def __call__(self, world, qs, qds):

        def C_collide(world, qs):
            collide_flags, mtvs, n_refs, p_refs, p_ins = self.collision_solver(world, qs)
            ux = n_refs - jnp.dot(n_refs.squeeze(), n_refs.squeeze())
            ux = ux / jnp.linalg.norm(ux)
            uy = jnp.cross(n_refs, ux)
            def depth(collide_flag, p_ref, p_in, vec):
                # return jnp.array([0])
                # print(collide_flag)
                # return jnp.where(collide_flag, jnp.dot(p_in-p_ref, vec), 0)
                # print('depth',jnp.dot(p_ref-p_in, vec))
                return jnp.where(collide_flag, jnp.dot(p_ref-p_in, vec), 0)
            
            
            # import pdb;pdb.set_trace()

            C_pen = jax.vmap(depth)(collide_flags, p_refs, p_ins, n_refs)
            return C_pen
            # C_fricx = jax.vmap(depth)(collide_flags, p_refs, p_ins, ux)
            # C_fricy = jax.vmap(depth)(collide_flags, p_refs, p_ins, uy)
            return jnp.concatenate([C_pen, C_fricx, C_fricy])
        def C_collide_d(world, qs, qds):
            J = jax.jacfwd(partial(C_collide, world))(qs)
            # import pdb;pdb.set_trace()
            J = J.reshape(1,N)
            # return (J.squeeze()*qds)
            # import pdb;pdb.set_trace()
            return (J.squeeze()@qds.reshape(N,1))
            # return (J*qds).sum()

            # Constraints

        J = jax.jacfwd(partial(C_collide, world), 0)(qs)
        J = J.reshape(J.shape[0],N)
        # JJ = jax.jacfwd(jax.jacfwd(partial(C_collide, world)))(qs)
        # JJ = jax.hessian(partial(C_collide, world))(qs)
        # JJ = JJ.reshape(N,N)
        # Jd = (JJ*qd_vec).sum(1)
        Jd = jax.jacfwd(partial(C_collide_d, world), 0)(qs, qds)
        Jd = Jd.reshape(1,N)
        K = J @ Minv @ J.T
        Kinv = jnp.linalg.inv(K)
        # print('J',J)

        
        # Lmult = Kinv @ J @ Minv @ (u_vec - C @ qd_vec - g)
        Lmult = Kinv @ J @ (Minv @ (u_vec - C @ qd_vec - g) + Jd@qd_vec)
        F = J.T @ -Lmult
        # JL = J.T @ -Lmult*2
        F = jnp.where(jnp.isnan(JL), 0, JL)

        return F