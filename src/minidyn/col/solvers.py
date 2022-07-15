from dataclasses import dataclass
import jax
from jax import numpy as jnp, random
import minidyn.dyn.functions as F
from jax.tree_util import register_pytree_node_class
from functools import partial
import trimesh
import minidyn as mdn
class Solver:
    pass

class SATSolver(Solver):

    def __init__(self):
        pass

    
    # @partial(jax.jit, static_argnums=(0,))
    def __call__(self, world, qs):
        colres = []

        for ((ib1, ib2), (b1, b2), spair)  in zip(world.body_pairs_idxs, world.body_pairs, world.shape_pairs):
            q1, q2 = qs[ib1], qs[ib2]
            b1, b2 = b1, b2
            res = []

            for (s1, s2) in spair:
                # import pdb;pdb.set_trace()
                res += [self.solve(q1, q2, s1, s2)]
            colres += [res]
        
        return colres
  
    def solve(self, q1, q2, s1, s2):
        
        v1 = F.vec2world(s1.vertices, F.q2tf(q1)) # for normals, zero translate
        n1 = F.vec2world(s1.face_normals, F.q2tf(jnp.array([*q1[:4], 0, 0, 0])))
        v2 = F.vec2world(s2.vertices, F.q2tf(q2))
        n2 = F.vec2world(s2.face_normals, F.q2tf(jnp.array([*q2[:4], 0, 0, 0])))

        naxes = jnp.concatenate([n1, n2], axis=0)
        def build_edge_vec(v):
            return F.vec_normalize(v - jnp.roll(v,-1,0))
            # return v - jnp.roll(v,-1,0)
        e1 = build_edge_vec(v1)
        e2 = build_edge_vec(v2)
        xaxes = jnp.reshape(jnp.cross(e1[:, jnp.newaxis, :], e2), (-1, 3))
        idxzeros = jnp.all(xaxes== 0, axis=1).reshape(-1, 1)
        replace = jnp.array([1,0,0]).tile((len(xaxes),1))
        # import pdb;pdb.set_trace()
        xaxes = jnp.where(idxzeros, replace, xaxes)
        # idxzeros = jnp.broadcast_to(idxzeros, (idxzeros.shape[0], 3))
        # @partial(jax.jit, static_argnums=(0,))

        # xaxes[idxzeros,:] = jnp.array([1.,0,0])
        # xaxes = xaxes[~jnp.all(xaxes== 0, axis=1)] # prune zero vectors
        xaxes = F.vec_normalize(xaxes)
        axes = jnp.concatenate([naxes, xaxes], axis=0)


        A_projs = axes @ v1.T
        B_projs = axes @ v2.T
        A_proj_maxs = jnp.max(A_projs, axis=1)
        A_proj_mins = jnp.min(A_projs, axis=1)
        B_proj_maxs = jnp.max(B_projs, axis=1)
        B_proj_mins = jnp.min(B_projs, axis=1)
        
        d1 = A_proj_mins < B_proj_maxs
        d2 = A_proj_maxs > B_proj_mins
        overlap= jnp.logical_and(d1, d2)
        # import pdb;pdb.set_trace()
        # print(d1andd2)

        # maxs = jnp.max(jnp.stack([A_proj_maxs, B_proj_maxs], axis=1), axis=1)
        # mins = jnp.min(jnp.stack([A_proj_mins, B_proj_mins], axis=1), axis=1)

        # d1 = (A_proj_maxs - A_proj_mins) + (B_proj_maxs - B_proj_mins)
        # d2 = (maxs - mins)
        # overlay = (d1 > d2)
        # assert (overlap==overlay).all()
        return overlap, overlap.all()

        # print(overlay)
        # import pdb;pdb.set_trace()
        return overlay, (overlay).all()

if __name__ == '__main__':
    solver = SATSolver()
    q1 = jnp.array([1, 0, 0, 0., 0, 0, 0]) 
    # q2 = jnp.array([1, 0, 0, 0., 0.99, 0.99, 0.99]) # collide
    # q2 = jnp.array([1, 0, 0, 0., 1.01, 0, 0]) # not collide
    q2 = jnp.array([1, 0, 0, 0., 0, 0, 10]) # not collide
    # s1 = mdn.col.Shape.from_trimesh(trimesh.creation.box((1., 1., 1.)))
    s1 = mdn.col.Shape.from_trimesh(trimesh.creation.box((10., 0.1, 10.)))
    s2 = mdn.col.Shape.from_trimesh(trimesh.creation.box((1., 1., 1.)))
    overlap, overlap_all = solver.solve(q1,q2,s1,s2)
    print(overlap)
    print(overlap_all)
    # import pdb;pdb.set_trace()
    