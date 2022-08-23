from typing import *
from functools import partial

import jax
from jax import numpy as jnp, random
from jax.tree_util import tree_map, register_pytree_node_class
from numpy import complexfloating
from zmq import has

from minidyn.dyn import integrators
from minidyn.dyn import collisions
from minidyn.dyn import contacts
from minidyn.dyn.contacts import RigidContact
from minidyn.dyn import world
from minidyn.dyn import functions as Fn

@register_pytree_node_class
class LagrangianDynamics(object):
    def __init__(self, dt=1/30., a=0):
        super().__init__()
        self.integrator = integrators.euler
        self.dt = dt
        self.a = a
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, world: world.World, qs, qds, us):            
        def get_L(q, qd, inertia, gravity):
            tf = Fn.q2tf(q)
            I = Fn.inertia_to_world(inertia, tf)
            v = Fn.qqd2v(q, qd)
            T = Fn.kinetic_energy(I, v)
            V = Fn.potential_energy(I, tf, gravity)
            return T - V, dict(T=T, V=V)
        
        # inertias = tree_map( lambda *values: 
        #         jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])
        inertias = Fn.stack_attr(world.bodies, 'inertia')
        is_static = jnp.tile(jnp.array(world.static_flags), (3,1)).T
        gravities = jnp.tile(world.gravity[jnp.newaxis,::], len(world.bodies)).reshape(len(world.bodies), 3)
        gravities = jnp.where(is_static == True, 0, gravities)
        
        Ms, energies = jax.vmap(jax.hessian(get_L, 1, True))(qs, qds, inertias, gravities)
        Ms = jnp.where(jnp.isnan(Ms), 0, Ms)
        Ms = Ms.squeeze((1, 2))
        Minvs = jnp.linalg.pinv(Ms, rcond=1e-20)
        gs = jax.vmap( jax.jacfwd(get_L, 0, True))(qs, qds, inertias, gravities)[0]
        gs = gs.squeeze((1, 2))
        # C = jax.jacfwd(jax.jacfwd(L, 1, True), 0)(q, qd, u, world)[0]
        def proto_get_forces(Minvs, us, gs, qds, J, Jd, C, body1_id, body2_id):
            Minv1, Minv2 = Minvs[body1_id], Minvs[body2_id]
            Minv = jnp.zeros([14, 14]).at[:7,:7].set(Minv1).at[7:,7:].set(Minv2)
            u = jnp.concatenate([us[body1_id], us[body2_id]])
            g = jnp.concatenate([gs[body1_id], gs[body2_id]])
            qd = jnp.concatenate([qds[body1_id], qds[body2_id]])
            K = J @ Minv @ J.T
            Kinv = jnp.linalg.pinv(K)
            T1 = J@Minv@(u-g)
            T2 = (Jd + 2*self.a*J)@qd
            T3 = (self.a**2)*C[::,jnp.newaxis]
            Lmult = Kinv @ (T1 + T2 + T3)

            F = J.T @ -Lmult
            # F1, F2 = jnp.split(F, 2, 0)
            F1, F2 = F[:7].squeeze(), F[7:].squeeze()
            Lmult1, Lmult2 = Lmult[:1], Lmult[1:]

            return dict(F1=F1, F2=F2, body1_id=body1_id, body2_id=body2_id, Lmult1=Lmult1, Lmult2=Lmult2)
        get_forces = partial(proto_get_forces, Minvs, us, gs, qds)

        F1_list, F2_list = [], []
        
        rigid_contacts_result = RigidContact.solve_world(world, qs, qds)
        (C_frics, C_pens, _, _, 
        J_frics, J_pens, Jd_frics, Jd_pens,
        _, body1_ids, body2_ids) = (v for (k, v) in rigid_contacts_result.items())

        pen_forces = jax.vmap(get_forces)(J_pens, Jd_pens, C_pens, body1_ids, body2_ids)
        F1_pens, F2_pens, _, _, body1_id_pens, body2_id_pens = (v for (k,v) in pen_forces.items())

        fric_forces = jax.vmap(get_forces)(J_frics, Jd_frics, C_frics, body1_ids, body2_ids)
        F1_frics, F2_frics, _, _, body1_id_frics, body2_id_frics = (v for (k,v) in fric_forces.items())

        F1s = jnp.concatenate([F1_pens, F1_frics])
        F2s = jnp.concatenate([F2_pens, F2_frics])
        F1_body_ids = jnp.concatenate([body1_id_pens, body1_id_frics])
        F2_body_ids = jnp.concatenate([body2_id_pens, body2_id_frics])

        def proto_map_F_to_qs_shape(qs_shape, F1, F2, F1_body_id, F2_body_id):
            return jnp.zeros(qs_shape).at[F1_body_id].set(F1).at[F2_body_id].set(F2)
        map_F_to_qs_shape = partial(proto_map_F_to_qs_shape, qs.shape)
        F_mapped = jax.vmap(map_F_to_qs_shape)(F1s, F2s, F1_body_ids, F2_body_ids)
        F_exts = jnp.sum(F_mapped, 0)

        def get_qdd(Minv, g, F_ext):
            return Minv @ (g - F_ext)
        qdds = jax.vmap(get_qdd)(Minvs, gs, F_exts)
        is_static= jnp.array(world.static_flags)[:, jnp.newaxis]
        qdds = jnp.where(is_static == True, 0, qdds)
        qs_new, qds_new = self.integrator(qs, qds, qdds, self.dt)

        # breakpoint()
        # jax.debug.print('qdd {x}',x=qdd_vec)
        # jax.debug.print('Minv {x}',x=Minv)
        # jax.debug.print('g {x}',x=g)
        # jax.debug.print('Fs {x}',x=Fs)
            #     jax.debug.print('joint {x}',x=joint)
    #     jax.debug.print('J {x}',x=J)
    #     jax.debug.print('Jd {x}',x=Jd)
    #     jax.debug.print('C {x}',x=C)
    #     jax.debug.print('Cd {x}',x=Cd)

        # jax.debug.print('colinfo {x}',x=col_info)


        # alpha = jnp.zeros(N)
        # alpha_pairs = tree_map( lambda *values: 
        #         jnp.stack(values, axis=0), 
        #         *[[t.shape_pair[0].alpha, t.shape_pair[1].alpha] 
        #             for t in world.rigid_contacts])
        # ib_pairs = tree_map( lambda *values: 
        #         jnp.stack(values, axis=0), *[t.ib_pair for t in world.rigid_contacts])
        # breakpoint()

        # dqd = (1+alpha)

        aux = dict(rigid_contacts_result, 
                    pen_forces=pen_forces, 
                    fric_forces=fric_forces,
                         energies=energies)
        return qs_new, qds_new, aux
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
