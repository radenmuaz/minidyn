from typing import *
from functools import partial

import jax
from jax import numpy as jnp, random
from jax import vmap, tree_map,lax
from jax.tree_util import register_pytree_node_class
from zmq import has

from minidyn.dyn import integrators
from minidyn.dyn import collisions
from minidyn.dyn import contacts
from minidyn.dyn import world
from minidyn.dyn import functions as F

@register_pytree_node_class
class LagrangianDynamics(object):
    def __init__(self, dt=1/30.):
        super().__init__()
        self.integrator = integrators.euler
        self.dt = dt
        self.collision_solver = collisions.SeparatingAxis()
        self.contact_solver = contacts.CompliantContacts()
    
    def __call__(self, world: world.World, q, qd, u):
        qdd, aux = self.solve(world, q, qd, u)
        q_new, qd_new = self.integrator(q, qd, qdd, self.dt)
        # N = q.size
        # q_vec = jnp.concatenate((q.reshape(N), qd.reshape(N)))
        # qd_vec = jnp.concatenate((qd.reshape(N), qdd.reshape(N)))
        # q_vec_new = self.integrator(q_vec, qd_vec, self.dt)
        # q_new = q_vec_new[:N].reshape(q.shape)
        # qd_new  = q_vec_new[N:].reshape(qd.shape)
        # import pdb;pdb.set_trace()
        def norm_quat_part(q):
            return q.at[:4].set(F.quat_norm(q[:4]))
        q_new = jax.vmap(norm_quat_part)(q_new)
        return q_new, qd_new, aux

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, world, q, qd, u):
        def get_energies(q, qd, u, world):
            tfs = vmap(F.q2tf)(q)
            Is_local = tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])

            Is = vmap(F.inertia_to_world)(Is_local, tfs)
            vs = vmap(F.qqd2v)(q, qd)
            Ts = vmap(F.kinetic_energy)(Is, vs)
            g = world.gravity.reshape(1,3)
            mask = jnp.array(world.static_masks).tile((3,1)).T
            gs = jnp.tile(g.reshape(1,3), len(tfs)).reshape(len(tfs), 3)
            gs = jnp.where(mask == True, 0, gs)
            Vs = vmap(F.potential_energy)(Is, tfs, gs)
            return Ts, Vs
        
        def L(q, qd, u, world):
            T, V = get_energies(q, qd, u, world)
            return jnp.sum(T) - jnp.sum(V), (T, V)
        
        # reference shape and q dot vector
        N = q.size
        q_vec = q.reshape(N, 1)
        qd_vec = qd.reshape(N, 1)
        u_vec = u.reshape(N, 1)

        # Mass, Coriolis, gravity
        M, (T, V) = jax.hessian(L, 1, True)(q, qd, u, world)
        M = M.reshape(N, N)
        M = jnp.where(jnp.isnan(M),0,M)
        Minv = jnp.linalg.pinv(M, rcond=1e-20)
        g = jax.grad(L, 0, True)(q, qd, u, world)[0]
        g = g.reshape(N, 1)
        C = jax.jacfwd(jax.grad(L, 1, True), 0)(q, qd, u, world)[0]
        C = C.reshape(N, N)
        C = jnp.where(jnp.isnan(C),0,C)
        # breakpoint()
        
        F_c, colaux = self.contact_solver(world, self.collision_solver, q, qd)
        F_c_vec = F_c.reshape(N, 1)

        qdd_vec = Minv @ ( g - C @ qd_vec + F_c_vec)
        qdd = qdd_vec.reshape(qd.shape)
        mask = jnp.array(world.static_masks)[:, jnp.newaxis]
        qdd = jnp.where(mask == True, 0, qdd)



        # jnp.set_printoptions(precision=4)
        # print(world.body_pairs_idxs)
        # col = self.collision_solver(world, q);print(col[0]); 
        # if col[0].any():import pdb;pdb.set_trace()

        aux = (F_c, colaux, g)#M, K, Lmult, J, Jd, JL)
        # import pdb;pdb.set_trace()
        return qdd, aux
    
    def tree_flatten(self):
        children = (
                    self.dt
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
