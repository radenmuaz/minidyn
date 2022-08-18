from typing import *
from functools import partial

import jax
from jax import numpy as jnp, random
from jax.tree_util import tree_map, register_pytree_node_class
from zmq import has

from minidyn.dyn import integrators
from minidyn.dyn import collisions
from minidyn.dyn import contacts
from minidyn.dyn.contacts import RigidContact
from minidyn.dyn import world
from minidyn.dyn import functions as F

@register_pytree_node_class
class LagrangianDynamics(object):
    def __init__(self, dt=1/30., a=0):
        super().__init__()
        self.integrator = integrators.euler
        self.dt = dt
        self.a = a
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, world: world.World, q, qd, u):
        

        def get_energies(q, qd, u, world):
            tfs = jax.vmap(F.q2tf)(q)
            Is_local = tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])

            Is = jax.vmap(F.inertia_to_world)(Is_local, tfs)
            vs = jax.vmap(F.qqd2v)(q, qd)
            Ts = jax.vmap(F.kinetic_energy)(Is, vs)
            g = world.gravity.reshape(1,3)
            # static = jnp.array(world.static_flags).tile((3,1)).T
            static = jnp.tile(jnp.array(world.static_flags), (3,1)).T
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
        def get_forces(J, Jd, C):
            J = J.reshape(J.shape[0],N)
            Jd = Jd.reshape(Jd.shape[0],N)
            C = C[::,jnp.newaxis]
            K = J @ Minv @ J.T
            Kinv = jnp.linalg.pinv(K)
            T1 = J@Minv@(u_vec-g)
            T2 = (Jd + 2*self.a*J)@qd_vec
            T3 = (self.a**2)*C
            Lmult = Kinv @ (T1 + T2 + T3)
            F_c = J.T @ -Lmult
            return dict(F_c=F_c, Lmult=Lmult)
            # return F_c, Lmult

        # Fs = []
        # Lmults = []
        # for joint in world.joints:
        #     Jd, J, Cd, C = joint(q, qd)
        #     F_j, Lmult_j = get_F_given_J_C(J, Jd, C)
        #     Fs += [F_j]
        #     Lmults += [Lmult_j]
        #     jax.debug.print('joint {x}',x=joint)
        #     jax.debug.print('J {x}',x=J)
        #     jax.debug.print('Jd {x}',x=Jd)
        #     jax.debug.print('C {x}',x=C)
        #     jax.debug.print('Cd {x}',x=Cd)

        rigid_contacts_result = RigidContact.vmap_solve(world.rigid_contacts, q, qd)
        # breakpoint()
        (C_fric, C_pen, Cd_fric, Cd_pen, 
        J_fric, J_pen, Jd_fric, Jd_pen,
            col_info) = (v for (k, v) in rigid_contacts_result.items())

        pen_forces = jax.vmap(get_forces)(J_pen, Jd_pen, C_pen)
        F_c_pen, Lmult_pen = (v for (k,v) in pen_forces.items())

        fric_forces = jax.vmap(get_forces)(J_fric, Jd_fric, C_fric)
        F_c_fric, Lmult_fric = (v for (k,v) in fric_forces.items())

        F_constr = jnp.sum(F_c_pen, 0) + jnp.sum(F_c_fric, 0) 
        qdd_vec = Minv @ (g - F_constr)
        qdd = qdd_vec.reshape(qd.shape)
        static= jnp.array(world.static_flags)[:, jnp.newaxis]
        qdd = jnp.where(static == True, 0, qdd)

        # jax.debug.print('qdd {x}',x=qdd_vec)
        # jax.debug.print('Minv {x}',x=Minv)
        # jax.debug.print('g {x}',x=g)
        # jax.debug.print('Fs {x}',x=Fs)

        aux = dict(rigid_contacts_result, 
                    pen_forces=pen_forces, 
                    fric_forces=fric_forces,
                         T=T, V=V)

        jax.debug.print('colinfo {x}',x=col_info)

        q_new, qd_new = self.integrator(q, qd, qdd, self.dt)

        alpha = jnp.zeros(N)
        alpha_pairs = tree_map( lambda *values: 
                jnp.stack(values, axis=0), 
                *[[t.shape_pair[0].alpha, t.shape_pair[1].alpha] 
                    for t in world.rigid_contacts])
        ib_pairs = tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[t.ib_pair for t in world.rigid_contacts])
        breakpoint()

        dqd = (1+alpha)

        
        return q_new, qd_new, aux
        # return qdd, aux
    
    def tree_flatten(self):
        children = (
                    self.dt,
                    self.a
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
