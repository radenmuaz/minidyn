from dataclasses import dataclass
import jax
from jax import numpy as jnp, random
import minidyn.dyn.functions as F

class Solver:
    pass

class SATSolver(Solver):
    def __init__(self):
        pass

    def __call__(self, world, qs):
        colres = []
        # qs_idxs = [[i, j] for i in range(len(world.bodies)) for j in range(len(world.bodies))]
        for idx, (ib1, ib2) in enumerate(world.body_pairs_idxs):
            q1, q2 = qs[ib1], qs[ib2]
            b1, b2 = world.bodies[ib1], world.bodies[ib2]
            colres += [[]]
            for (is1, is2) in world.shape_pairs_idxs[idx]:
                colres[idx] += self.sat(q1, q2, b1.shapes[is1], b2.shapes[is2])
            
    def sat(self, q1, q2, s1, s2):
        # def v2tf(v):
        #     return jnp.array([[1, 0, 0, v[0]],
        #                       [0, 1, 0, v[1]],
        #                       [0, 0, 1, v[2]],
        #                       [0, 0, 0, 0]])
        # def tf2v(tf):
        #     return tf[:3,3]
        def v2tf(v): #[1,2,3] -> [1,2,3,0]
            return jnp.concatenate((v, jnp.array((0,)))).reshape(4,1)
        def tf2v(tf): # [1,2,3,0] -> [1,2,3]
            return tf[:3] 
        Nv1, Nv2 = s1.vertices.shape[0], s2.vertices.shape[0]
        tf_q1 = jnp.broadcast_to(F.q2tf(q1), (Nv1, 4, 4))
        tf_q2 = jnp.broadcast_to(F.q2tf(q2), (Nv2, 4, 4))
        tf_v1, tf_v2 = jax.vmap(v2tf)(s1.vertices), jax.vmap(v2tf)(s2.vertices)
        tf_q1v1, tf_q2v2 = tf_q1 @ tf_v1, tf_q2 @ tf_v2
        v1, v2 = jax.vmap(tf2v)(tf_q1v1), jax.vmap(tf2v)(tf_q2v2)
        # v1, v2 = 
        v1s, v2s = jnp.roll(v1, -1, 0), jnp.roll(v2, -1, 0)
        e1, e2 =  v1s - v1, v2s - v2
        import pdb;pdb.set_trace()
        return
    
            