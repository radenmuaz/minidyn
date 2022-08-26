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
    
    # @partial(jax.jit, static_argnums=(0,))
    @jax.jit
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
        is_static= jnp.array(world.static_flags)[:, jnp.newaxis]
        # is_static = jnp.tile(jnp.array(world.static_flags), (3,1)).T
        gravities = jnp.tile(world.gravity[jnp.newaxis,::], len(world.bodies)).reshape(len(world.bodies), 3)
        gravities = jnp.where(is_static == True, 0, gravities)
        
        Ms, energies = jax.vmap(jax.hessian(get_L, 1, True))(qs, qds, inertias, gravities)
        Ms = jnp.where(jnp.isnan(Ms), 0, Ms)
        Ms = Ms.squeeze((1, 2))
        Minvs = jnp.linalg.pinv(Ms, rcond=1e-20)
        gs = jax.vmap( jax.jacfwd(get_L, 0, True))(qs, qds, inertias, gravities)[0]
        gs = gs.squeeze((1, 2))
        # Coriolis_forces = jax.jacfwd(jax.jacfwd(L, 1, True), 0)(q, qd, u, world)[0]

        def proto_get_forces(Minvs, us, gs, qds, Lmult_func, Lmult_args, J, Jd, C, body1_id, body2_id):
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
            Lmult = Lmult_func(Lmult, Lmult_args)

            F = J.T @ -Lmult
            F1, F2 = F[:7].squeeze(), F[7:].squeeze()

            return dict(F1=F1, F2=F2, body1_id=body1_id, body2_id=body2_id, Lmult=Lmult)
        
        F1_list, F2_list = [], []
        body1_id_list, body2_id_list = [], []
        constraints_results = dict()
        forces = dict()

        if (len(world.rigid_contacts)>0):
            rigid_contacts_result = RigidContact.solve_world_d(world, qs, qds)
            rcr = rigid_contacts_result
            C_fric1s, C_fric2s, C_pens = rcr['C_fric1'], rcr['C_fric1'], rcr['C_pen']
            J_fric1s, J_fric2s, J_pens = rcr['J_fric1'], rcr['J_fric2'], rcr['J_pen']
            Jd_fric1s, Jd_fric2s, Jd_pens = rcr['Jd_fric1'], rcr['Jd_fric2'], rcr['Jd_pen']
            body1_ids, body2_ids = rcr['body1_id'], rcr['body2_id']
            mu1s, mu2s= rcr['mu1'], rcr['mu2']
            jax.debug.print('J_pens {x}',x=J_pens)
            jax.debug.print('col_info {x}',x=rigid_contacts_result['col_info'])

            def nothing(Lmult, Lmult_args):
                return Lmult
            get_forces_pen = partial(proto_get_forces, Minvs, us, gs, qds, nothing, None)
            pen_forces = jax.vmap(get_forces_pen)(J_pens, Jd_pens, C_pens, body1_ids, body2_ids)
            pf = pen_forces
            F1_pens, F2_pens, body1_id_pens, body2_id_pens = pf['F1'], pf['F2'], pf['body1_id'], pf['body2_id']
            def Lmult_fric(Lmult, Lmult_args):
                mu, F_n = Lmult_args
                mag_F_n = jnp.linalg.norm(F_n)
                Lmult_new = jnp.clip(Lmult, -mu*mag_F_n, mu*mag_F_n)
                return Lmult_new
        
            get_forces_friction = partial(proto_get_forces, Minvs, us, gs, qds, Lmult_fric)

            fric1_forces = jax.vmap(get_forces_friction)((mu1s, F1_pens), J_fric1s, Jd_fric1s, C_fric1s, body1_ids, body2_ids)
            f1f = fric1_forces
            F1_fric1s, F2_fric1s, body1_id_fric1s, body2_id_fric1s = f1f['F1'], f1f['F2'], f1f['body1_id'], f1f['body2_id']

            
            fric2_forces = jax.vmap(get_forces_friction)((mu2s, F2_pens),J_fric2s, Jd_fric2s, C_fric2s, body1_ids, body2_ids)
            f2f = fric2_forces
            F1_fric2s, F2_fric2s, body1_id_fric2s, body2_id_fric2s = f2f['F1'], f2f['F2'], f2f['body1_id'], f2f['body2_id']

            F1_list += [F1_pens, F1_fric1s, F1_fric2s]
            F2_list += [F2_pens, F2_fric1s, F2_fric2s]
            body1_id_list += [body1_id_pens, body1_id_fric1s, body1_id_fric2s]
            body2_id_list += [body2_id_pens, body2_id_fric1s, body2_id_fric2s]

            constraints_results['rigid_contacts'] = rigid_contacts_result
            forces['pen'] = pen_forces
            forces['fric1'] = fric1_forces
            forces['fric2'] = fric2_forces

        if (len(F1_list)>0):
            F1s, F2s = jnp.concatenate(F1_list), jnp.concatenate(F2_list)
            body1_ids, body2_ids = jnp.concatenate(body1_id_list), jnp.concatenate(body2_id_list)

            def proto_map_F_to_qs_shape(qs_shape, F1, F2, body1_id, body2_id):
                return jnp.zeros(qs_shape).at[body1_id].set(F1).at[body2_id].set(F2)
            map_F_to_qs_shape = partial(proto_map_F_to_qs_shape, qs.shape)
            F_mapped = jax.vmap(map_F_to_qs_shape)(F1s, F2s, body1_ids, body2_ids)
            F_exts = jnp.sum(F_mapped, 0)
        else:
            F_exts = jnp.zeros_like(qs)

        def get_qdd(Minv, g, F_ext):
            return Minv @ (g - F_ext)
        qdds = jax.vmap(get_qdd)(Minvs, gs, F_exts)
        qdds = jnp.where(is_static == True, 0, qdds)
        qs_new, qds_new = self.integrator(qs, qds, qdds, self.dt)

        breakpoint()
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

        if (len(world.rigid_contacts)>0):
            alpha1s, alpha2s= rcr['alpha1'], rcr['alpha2']

        # alpha = jnp.zeros(N)
        # alpha_pairs = tree_map( lambda *values: 
        #         jnp.stack(values, axis=0), 
        #         *[[t.shape_pair[0].alpha, t.shape_pair[1].alpha] 
        #             for t in world.rigid_contacts])
        # ib_pairs = tree_map( lambda *values: 
        #         jnp.stack(values, axis=0), *[t.ib_pair for t in world.rigid_contacts])
        # breakpoint()

        # dqd = (1+alpha)

        aux = dict(constraints_results=constraints_results,
                    forces=forces,
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
