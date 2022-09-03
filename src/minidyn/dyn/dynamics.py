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
from minidyn.dyn.joints import FixedJoint
from minidyn.dyn import world
from minidyn.dyn import functions as Fn

@register_pytree_node_class
class LagrangianDynamics(object):
    def __init__(self, dt=1/30., a=0.01):
        super().__init__()
        self.integrator = integrators.euler
        self.dt = dt
        self.a = a
    
    @partial(jax.jit, static_argnums=(0,))
    # @jax.jit
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
        # Fn.kinetic_energy(world.bodies[0].inertia, Fn.qqd2v(qs[0], qds[0]))
        # Fn.potential_energy(Fn.inertia_to_world(world.bodies[0].inertia, Fn.q2tf(qs[0])), Fn.q2tf(qs[0]), gravities[0])

        # jax.hessian(get_L,1,True)(qs[0], qds[0], world.bodies[0].inertia, gravities[0])
        
        Ms, energies = jax.vmap(jax.hessian(get_L, 1, True))(qs, qds, inertias, gravities)
        # print(energies)
        # print(Ms)
        Ms = jnp.where(jnp.isnan(Ms), 0, Ms)
        # Ms = Ms.squeeze((1, 2))
        Minvs = jnp.linalg.pinv(Ms, rcond=1e-20)
        gs = jax.vmap( jax.jacfwd(get_L, 0, True))(qs, qds, inertias, gravities)[0]
        # gs = gs.squeeze((1, 2))
        # Coriolis_forces = jax.jacfwd(jax.jacfwd(L, 1, True), 0)(q, qd, u, world)[0]

        def proto_get_forces(Minvs, us, gs, qds, Lmult_func, Lmult_args, J, Jd, C, body1_id, body2_id):
            Minv1, Minv2 = Minvs[body1_id], Minvs[body2_id]
            Minv = jnp.zeros([14, 14]).at[:7,:7].set(Minv1).at[7:,7:].set(Minv2)
            u = jnp.concatenate([us[body1_id], us[body2_id]])
            g = jnp.concatenate([gs[body1_id], gs[body2_id]])
            qd = jnp.concatenate([qds[body1_id], qds[body2_id]])
            K = J @ Minv @ J.T
            Kinv = 1/K
            Kinv = jnp.where(jnp.isinf(Kinv), 0, Kinv)
            # Kinv = jnp.where(jnp.isinf(Kinv), 1e9, Kinv)
            # breakpoint()
            T1 = J@Minv@(u-g)
            T2 = (Jd + 2*self.a*J)@qd
            T3 = (self.a**2)*C
            # T3 = (self.a**2)*C[::,jnp.newaxis]
            Lmult = Kinv * (T1 - T2 - T3)
            if Lmult_func is not None:
                Lmult = Lmult_func(Lmult, Lmult_args)

            F = J * -Lmult
            F1, F2 = F[:7], F[7:]

            return dict(F=F, F1=F1, F2=F2, J=J, Lmult=Lmult, body1_id=body1_id, body2_id=body2_id, )
        
        

        F1_list, F2_list = [], []
        body1_id_list, body2_id_list = [], []
        constraints_results = dict()
        forces = dict()

        if (len(world.joints)>0):
            # breakpoint()
            fixed_joints_result = FixedJoint.solve_world_d(world, qs, qds)
            fjr = fixed_joints_result
            C, J, Jd, dCddqd = fjr['C'], fjr['J'], fjr['Jd'], fjr['dCddqd']
            N = C.shape[1]
            body1_ids = jnp.repeat(fjr['body1_id'], N)[jnp.newaxis,::]
            body2_ids = jnp.repeat(fjr['body2_id'], N)[jnp.newaxis,::]

            get_forces = partial(proto_get_forces, Minvs, us, gs, qds, None, None)
            # jax.vmap(get_forces)(J[0], Jd[0], C[0], body1_ids[0], body2_ids[0])
            forces = jax.vmap(jax.vmap(get_forces))(J, Jd, C, body1_ids, body2_ids)
            # breakpoint()

            F1_list += [forces['F1']]
            F2_list += [forces['F2']]
            body1_id_list += [forces['body1_id']]
            body2_id_list += [forces['body2_id']]
            jax.debug.print('F1 {x}',x=forces['F1'])
            jax.debug.print('C {x}',x=C)
            # jax.debug.print('J {x}',x=J)



        if (len(world.rigid_contacts)>0):
            rigid_contacts_result = RigidContact.solve_world_d(world, qs, qds)
            rcr = rigid_contacts_result
            C_fric1s, C_fric2s, C_pens = rcr['C_fric1'].squeeze(1), rcr['C_fric1'].squeeze(1), rcr['C_pen'].squeeze(1)
            J_fric1s, J_fric2s, J_pens = rcr['J_fric1'].squeeze(1), rcr['J_fric2'].squeeze(1), rcr['J_pen'].squeeze(1)
            Jd_fric1s, Jd_fric2s, Jd_pens = rcr['Jd_fric1'].squeeze(1), rcr['Jd_fric2'].squeeze(1), rcr['Jd_pen'].squeeze(1)
            # body1_ids, body2_ids = rcr['body1_id'], rcr['body2_id']
            body1_ids = rcr['body1_id']
            body2_ids = rcr['body2_id']
            mu1s = rcr['mu1']
            mu2s = rcr['mu2']
            jax.debug.print('J_pens {x}',x=J_pens)
            jax.debug.print('col_info {x}',x=rigid_contacts_result['col_info'])

            get_forces_pen = partial(proto_get_forces, Minvs, us, gs, qds, None, None)
            pen_forces = jax.vmap(get_forces_pen)(J_pens, Jd_pens, C_pens, body1_ids, body2_ids)
            pf = pen_forces
            F_pens = pf['F']
            F1_pens, F2_pens,= pf['F1'], pf['F2']
            body1_id_pens, body2_id_pens = pf['body1_id'], pf['body2_id']
            
            def Lmult_fric(Lmult, Lmult_args):
                mu, F_n = Lmult_args
                mag_F_n = jnp.linalg.norm(F_n)
                Lmult_new = jnp.clip(Lmult, -mu*mag_F_n, mu*mag_F_n)
                return Lmult_new
        
            get_forces_friction = partial(proto_get_forces, Minvs, us, gs, qds, Lmult_fric)
            # breakpoint()
            fric1_forces = jax.vmap(get_forces_friction)((mu1s, F_pens), J_fric1s, Jd_fric1s, C_fric1s, body1_ids, body2_ids)
            f1f = fric1_forces
            F1_fric1s, F2_fric1s, body1_id_fric1s, body2_id_fric1s = f1f['F1'], f1f['F2'], f1f['body1_id'], f1f['body2_id']

            fric2_forces = jax.vmap(get_forces_friction)((mu1s, F_pens), J_fric2s, Jd_fric2s, C_fric2s, body1_ids, body2_ids)
            # fric2_forces = jax.vmap(get_forces_friction)((mu2s, F_pens),J_fric2s, Jd_fric2s, C_fric2s, body1_ids, body2_ids)
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

            did_collides = rcr['col_info']['did_collide']
            alpha1s, alpha2s= rcr['alpha1'], rcr['alpha2']
            Cd_pens = rcr['Cd_pen'].squeeze(1)
            J_pens = pf['J']
            def get_Minv_from_ids(Minvs, body1_id, body2_id):
                Minv1, Minv2 = Minvs[body1_id], Minvs[body2_id]
                Minv = jnp.zeros([14, 14]).at[:7,:7].set(Minv1).at[7:,7:].set(Minv2)
                return Minv
            Minv_pens = jax.vmap(partial(get_Minv_from_ids, Minvs))(body1_ids, body2_ids)
            def get_qd_from_ids(qds, body1_id, body2_id):
                qd1, qd2 = qds[body1_id], qds[body2_id]
                qd = jnp.concatenate([qd1, qd2])
                return qd
            qd_pens = jax.vmap(partial(get_qd_from_ids, qds))(body1_ids, body2_ids)

            def get_dqd_pen(Minv, qd, J, did_collide, Cd, alpha1, alpha2):
                def bounce(Minv, qd, J, alpha1, alpha2):
                    # J = J[:,jnp.newaxis]
                    alpha = jnp.max(jnp.stack([alpha1, alpha2]))
                    K = J @ Minv @ J.T
                    Kinv = 1/K
                    Kinv = jnp.where(jnp.isinf(Kinv), 0, Kinv)
                    dqd = (1+alpha)*(Minv@J.T*Kinv*J@qd)
                    # dqd = (1+alpha)*(Minv@J.T@jnp.linalg.pinv(J@Minv@J.T)@J@qd)
                    # dqd = (1+alpha)*(Minv@J.T@jnp.linalg.pinv((J@Minv@J.T).reshape(1, 1))@J@qd)

                    return dqd
                do_apply = jnp.logical_and(did_collide, Cd >= 0)
                return jnp.where(do_apply, 
                        bounce(Minv, qd, J, alpha1, alpha2),
                        jnp.zeros(14))
            dqd_pens = jax.vmap(get_dqd_pen)(Minv_pens, qd_pens, J_pens,did_collides, Cd_pens, alpha1s, alpha2s)
            # breakpoint()
            # if did_collides.any():
            #     breakpoint()
            dqd_pen1s, dqd_pen2s = jnp.split(dqd_pens, 2, 1)
            def proto_map_dqd_to_qs_shape(qs_shape, qd1, qd2, body1_id, body2_id):
                return jnp.zeros(qs_shape).at[body1_id].set(qd1).at[body2_id].set(qd2)
            map_dqd_to_qs_shape = partial(proto_map_dqd_to_qs_shape, qs.shape)
            dqd_pens_mapped = jax.vmap(map_dqd_to_qs_shape)(dqd_pen1s, dqd_pen2s, body1_ids, body2_ids)
            jax.debug.print('Cd_pens {x}',x=Cd_pens)
            jax.debug.print('dqd_pens {x}',x=dqd_pens)
        else:
            dqd_pens_mapped = jnp.zeros_like(qds)

        def proto_get_velocity_forces(Minvs, us, gs, qds, Lmult_func, Lmult_args, Jd, dCddqd, Cd, body1_id, body2_id):
            Minv1, Minv2 = Minvs[body1_id], Minvs[body2_id]
            Minv = jnp.zeros([14, 14]).at[:7,:7].set(Minv1).at[7:,7:].set(Minv2)
            u = jnp.concatenate([us[body1_id], us[body2_id]])
            g = jnp.concatenate([gs[body1_id], gs[body2_id]])
            qd = jnp.concatenate([qds[body1_id], qds[body2_id]])
            K = dCddqd @ Minv @ dCddqd.T
            Kinv = jnp.linalg.pinv(K)
            T1 = dCddqd@Minv@(u-g)
            T2 = Jd@qd
            # T3 = (self.a)*Cd[::,jnp.newaxis]
            T3 = self.a*Cd[::,jnp.newaxis]
            Lmult = -Kinv @ (T1 + T2 + T3)
            if Lmult_func is not None:
                Lmult = Lmult_func(Lmult, Lmult_args)
            jax.debug.print('Cd {x}',x=Cd)

            # F = dCddqd.T @ -Lmult
            F = dCddqd.T @ Lmult
            F1, F2 = F[:7].squeeze(), F[7:].squeeze()

            return dict(F=F, F1=F1, F2=F2, dCddqd=dCddqd, Lmult=Lmult, body1_id=body1_id, body2_id=body2_id, )
        
        # if (len(world.rigid_contacts)>0):
        #     rigid_contacts_result = RigidContact.solve_world_d(world, qs, qds)
        #     rcr = rigid_contacts_result
        #     Cd_fric1s, Cd_fric2s, Cd_pens = rcr['Cd_fric1'], rcr['Cd_fric1'], rcr['Cd_pen']
        #     dCddqd_fric1s, dCddqd_fric2s, dCddqd_pens = rcr['dCddqd_fric1'], rcr['dCddqd_fric2'], rcr['dCddqd_pen']
        #     Jd_fric1s, Jd_fric2s, Jd_pens = rcr['Jd_fric1'], rcr['Jd_fric2'], rcr['Jd_pen']
        #     body1_ids, body2_ids = rcr['body1_id'], rcr['body2_id']
        #     mu1s, mu2s= rcr['mu1'], rcr['mu2']
        #     jax.debug.print('dCddqd_pens {x}',x=dCddqd_pens)
        #     jax.debug.print('col_info {x}',x=rigid_contacts_result['col_info'])

        #     get_forces_pen = partial(proto_get_velocity_forces, Minvs, us, gs, qds, None, None)
        #     pen_forces = jax.vmap(get_forces_pen)(Jd_pens, dCddqd_pens, Cd_pens, body1_ids, body2_ids)
        #     pf = pen_forces
        #     F_pens = pf['F']
        #     F1_pens, F2_pens,= pf['F1'], pf['F2']
        #     body1_id_pens, body2_id_pens = pf['body1_id'], pf['body2_id']
        #     jax.debug.print('F {x}',x=F_pens)
            
        #     def Lmult_fric(Lmult, Lmult_args):
        #         mu, F_n = Lmult_args
        #         mag_F_n = jnp.linalg.norm(F_n)
        #         Lmult_new = jnp.clip(Lmult, -mu*mag_F_n, mu*mag_F_n)
        #         return Lmult_new
        
        #     get_forces_friction = partial(proto_get_velocity_forces, Minvs, us, gs, qds, Lmult_fric)

        #     fric1_forces = jax.vmap(get_forces_friction)((mu1s, F_pens), Jd_fric1s, dCddqd_fric1s, Cd_fric1s, body1_ids, body2_ids)
        #     f1f = fric1_forces
        #     F1_fric1s, F2_fric1s, body1_id_fric1s, body2_id_fric1s = f1f['F1'], f1f['F2'], f1f['body1_id'], f1f['body2_id']

        #     fric2_forces = jax.vmap(get_forces_friction)((mu2s, F_pens),Jd_fric2s, dCddqd_fric2s, Cd_fric2s, body1_ids, body2_ids)
        #     f2f = fric2_forces
        #     F1_fric2s, F2_fric2s, body1_id_fric2s, body2_id_fric2s = f2f['F1'], f2f['F2'], f2f['body1_id'], f2f['body2_id']

        #     F1_list += [F1_pens, F1_fric1s, F1_fric2s]
        #     F2_list += [F2_pens, F2_fric1s, F2_fric2s]
        #     body1_id_list += [body1_id_pens, body1_id_fric1s, body1_id_fric2s]
        #     body2_id_list += [body2_id_pens, body2_id_fric1s, body2_id_fric2s]

        #     constraints_results['rigid_contacts'] = rigid_contacts_result
        #     forces['pen'] = pen_forces
        #     forces['fric1'] = fric1_forces
        #     forces['fric2'] = fric2_forces

        #     dqd_pens_mapped = jnp.zeros_like(qds)

        if (len(F1_list)>0):
            # breakpoint()
            F1_list_flat = [f.reshape(-1, 7) for f in F1_list]
            F2_list_flat = [f.reshape(-1, 7) for f in F2_list]
            body1_id_list_flat = [b.reshape(-1) for b in body1_id_list]
            body2_id_list_flat = [b.reshape(-1) for b in body2_id_list]
            # breakpoint()
            F1s, F2s = jnp.concatenate(F1_list_flat), jnp.concatenate(F2_list_flat)
            body1_ids, body2_ids = jnp.concatenate(body1_id_list_flat), jnp.concatenate(body2_id_list_flat)

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
        qds_new = qds_new - jnp.sum(dqd_pens_mapped, 0)    

        aux = dict(constraints_results=constraints_results,
                    forces=forces,
                         energies=energies)    
        # breakpoint()
        
        return qs_new, qds_new, aux
    
    def debug_print(aux):
        jax.debug.print('qdd {x}',x=qdd_vec)
        jax.debug.print('Minv {x}',x=Minv)
        jax.debug.print('g {x}',x=g)
        jax.debug.print('Fs {x}',x=Fs)
        jax.debug.print('joint {x}',x=joint)
        jax.debug.print('J {x}',x=J)
        jax.debug.print('Jd {x}',x=Jd)
        jax.debug.print('C {x}',x=C)
        jax.debug.print('Cd {x}',x=Cd)

        jax.debug.print('colinfo {x}',x=col_info)
    
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
