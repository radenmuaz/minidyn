import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
from minidyn.dyn.body import Body
import minidyn.dyn.functions as F
import networkx
from minidyn.dyn.spatial import Inertia
from minidyn.dyn import integrators 
from typing import *
from functools import partial
from jax import vmap, tree_map,lax
from jax.tree_util import register_pytree_node_class

# functions = [func1, func2, func3]
# index = jnp.arange(len(functions))
# x = jnp.ones((3, 5))

# vmap_functions = vmap(lambda i, x: lax.switch(i, functions, x))
# vmap_functions(index, x)

@register_pytree_node_class
class World: 
    '''
    Bodies connected with joints
    '''
    def __init__(self, root=Body(), joints=[],
                     bodies=[], gravity=jnp.array((0, 0, -9.81)),
                    init_qs=[], init_qds=[]):
        self.root = root
        # self.graph = networkx.DiGraph()
        # self.graph.add_node(self.root)
        self.joints = joints
        self.bodies = bodies
        # self.body2id = {self.root: -1}
        self.gravity = gravity
        self.init_qs = init_qs
        self.init_qds = init_qds

    def add_body(self, body, q=None, qd=None):
        self.bodies += [body,]
        # self.body2id[body] = len(self.body2id)
        q = q if q is not None else jnp.zeros(7).at[0].set(1)
        qd = qd if qd is not None else jnp.zeros(7).at[0].set(1e-9)
        self.init_qs += [q]
        # self.init_qs += [jnp.zeros(7).at[0].set(1).at[6].set(10)]
        self.init_qds += [qd]
        # self.init_qds += [jnp.zeros(7).at[0].set(1e-9).at[1].set(1)]
        # self.graph.add_node(body)
    
    def add_joint(self, joint, body, pred_body):
        self.joints += [joint,]
        # self.graph.add_edge(body, pred_body, joint=joint)
    
    def get_init_state(self):
        return jnp.vstack(self.init_qs), jnp.vstack(self.init_qds)
    
    def tree_flatten(self):
        children = (
                    self.root,
                    self.joints,
                    self.bodies,
                    # self.body2id,
                    self.gravity,
                    self.init_qs,
                    self.init_qds
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    

@register_pytree_node_class
class Sim(object):
    def __init__(self, dt=1e-3):
        super().__init__()
        self.integrator = integrators.euler
        self.dt = dt
    
    def __call__(self, world: World, qs, qds, u):
        qdds = self.solve(world, qs, qds, u)
        N = qs.size
        q_vec = cat((qs.reshape(N), qds.reshape(N)))
        qd_vec = cat((qds.reshape(N), qdds.reshape(N)))
        q_vec_new = self.integrator(q_vec, qd_vec, self.dt)
        qs_new = q_vec_new[:N].reshape(qs.shape)
        qds_new  = q_vec_new[N:].reshape(qds.shape)
        # import pdb;pdb.set_trace()
        return qs_new, qds_new

    # @jax.jit
    # def get_energies(self, qs, qds, u, world):
    #     tfs = vmap(F.q2tf)(qs)
    #     # Is_local = [body.inertia for body in world.bodies]
    #     Is_local = tree_map( lambda *values: 
    #         jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])

    #     Is = vmap(F.inertia_to_world)(Is_local, tfs)
    #     vs = vmap(F.qqd2v)(qs, qds)
    #     Ts = vmap(F.kinetic_energy)(Is, vs)
    #     gs = jnp.tile(world.gravity.reshape(1,3), len(tfs))
    #     Vs = vmap(F.potential_energy)(Is, tfs, gs)
    #     return Ts, Vs
    @partial(jax.jit, static_argnums=(0,))
    def solve(self, world, qs, qds, u):
        def get_energies(qs, qds, u, world):
            tfs = vmap(F.q2tf)(qs)
            Is_local = tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])

            Is = vmap(F.inertia_to_world)(Is_local, tfs)
            vs = vmap(F.qqd2v)(qs, qds)
            Ts = vmap(F.kinetic_energy)(Is, vs)
            gs = jnp.tile(world.gravity.reshape(1,3), len(tfs))
            Vs = vmap(F.potential_energy)(Is, tfs, gs)
            return Ts, Vs
        def L(qs, qds, u, world):
            Ts, Vs = get_energies(qs, qds, u, world)
            # Ts, Vs = self.get_energies(qs, qds, u, world)
            return jnp.sum(Ts) - jnp.sum(Vs)
        N = qs.size
        M = jax.hessian(L, 1)(qs, qds, u, world).reshape(N, N)
        Minv = jnp.linalg.pinv(M, rcond=1e-20)
        G = jax.grad(L, 0)(qs, qds, u, world).reshape(N, 1)
        C =  jax.jacfwd(jax.grad(L, 1), 0)(qs, qds, u, world).reshape(N, N)
        qd_vec = qds.reshape(N, 1)
        qdds_vec = Minv @ ( G - C @ qd_vec)
        qdds = qdds_vec.reshape(qds.shape)
        return qdds
    
    def tree_flatten(self):
        children = (
                    self.dt
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    