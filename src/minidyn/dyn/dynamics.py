from typing import *
from functools import partial

import jax
from jax import numpy as jnp, random
from jax.tree_util import tree_map, register_pytree_node_class
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
        # def norm_quat_part(q): 
        #     return q.at[:4].set(F.quat_norm(q[:4]))
        # q_new = jax.vmap(norm_quat_part)(q_new)
        return q_new, qd_new, aux

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, world, q, qd, u):
        def get_energies(q, qd, u, world):
            tfs = jax.vmap(F.q2tf)(q)
            Is_local = tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])

            Is = jax.vmap(F.inertia_to_world)(Is_local, tfs)
            vs = jax.vmap(F.qqd2v)(q, qd)
            Ts = jax.vmap(F.kinetic_energy)(Is, vs)
            g = world.gravity.reshape(1,3)
            static = jnp.array(world.static_flags).tile((3,1)).T
            gs = jnp.tile(g.reshape(1,3), len(tfs)).reshape(len(tfs), 3)
            gs = jnp.where(static == True, 0, gs)
            Vs = jax.vmap(F.potential_energy)(Is, tfs, gs)
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
        # C = jax.jacfwd(jax.grad(L, 1, True), 0)(q, qd, u, world)[0]
        # C = C.reshape(N, N)
        # C = jnp.where(jnp.isnan(C),0,C)
        # breakpoint()

        Fs = []
        for joint in world.joints:
            Jdo, Jo, Cdo, Co = joint(q, qd)
            J = Jo.reshape(Jo.shape[0],N)
            Jd = Jdo.reshape(Jdo.shape[0],N)
            C = Co[::,jnp.newaxis]
            Cd = Cdo[::,jnp.newaxis]
            # J = J.reshape(N, J.shape[0]).T


            K = J @ Minv @ J.T
            Kinv = jnp.linalg.pinv(K)
            # Kinv = jnp.linalg.inv(K+jnp.eye(7,7)*1e-9)
            a = 1
            Lmult = Kinv @ (J@Minv@(u_vec-g) + (Jd@qd_vec))
            # breakpoint()
            # Lmult = Kinv @ (J@Minv@(u_vec-g) + (Jd+2*a*J)@qd_vec + (a^2)*C)
            jax.debug.print('J {x}',x=Jo)
            jax.debug.print('Jd {x}',x=Jdo)
            jax.debug.print('Cd {x}',x=Cdo)
            # if (Jd@qd_vec).sum()>0:breakpoint()
            # def true(x): jax.debug.breakpoint()
            # def false(x): pass
            # jax.lax.cond((Jd@qd_vec).sum()>0, true, false, None)
            F_c = J.T @ -Lmult
            Fs += [F_c]
            # F_c = jnp.where(jnp.isnan(F_c), 0, F_c)

        
        F_c, colaux = self.contact_solver(world, self.collision_solver, q, qd)
        F_c_vec = F_c.reshape(N, 1)
        # F_c_vec = jnp.zeros((N, 1))
        Fs += [F_c_vec]

        qdd_vec = Minv @ (g - sum(Fs))
        jax.debug.print('qdd {x}',x=qdd_vec)
        jax.debug.print('Minv {x}',x=Minv)
        jax.debug.print('g {x}',x=g)
        jax.debug.print('Fs {x}',x=Fs)
        # qdd_vec = Minv @ ( g - C @ qd_vec - F_c_vec)
        # qdd_vec = Minv @ ( g - C @ qd_vec + F_c_vec)
        qdd = qdd_vec.reshape(qd.shape)
        static= jnp.array(world.static_flags)[:, jnp.newaxis]
        qdd = jnp.where(static == True, 0, qdd)



        # jnp.set_printoptions(precision=4)
        # print(world.body_pairs_idxs)
        # col = self.collision_solver(world, q);print(col[0]); 
        # if col[0].any():import pdb;pdb.set_trace()

        aux = (F_c, colaux, g, T, V)#M, K, Lmult, J, Jd, JL)
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
