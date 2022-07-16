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
        # axes = jnp.concatenate([naxes, xaxes], axis=0)


        np1 = naxes @ v1.T
        np2 = naxes @ v2.T
        np1_maxs = jnp.max(np1, axis=1)
        np1_mins = jnp.min(np1, axis=1)
        np2_maxs = jnp.max(np2, axis=1)
        np2_mins = jnp.min(np2, axis=1)

        xp1 = xaxes @ v1.T
        xp2 = xaxes @ v2.T
        xp1_maxs = jnp.max(xp1, axis=1)
        xp1_mins = jnp.min(xp1, axis=1)
        xp2_maxs = jnp.max(xp2, axis=1)
        xp2_mins = jnp.min(xp2, axis=1)
        
        is_n_overlap = jnp.logical_and(np1_mins < np2_maxs, np1_maxs > np2_mins)
        is_x_overlap = jnp.logical_and(xp1_mins < xp2_maxs, xp1_maxs > xp2_mins)
        is_overlap = jnp.logical_and(is_n_overlap.all(), is_x_overlap.all())
        def did_overlap(naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs):
            n_left = jnp.abs(np1_mins - np2_maxs)
            n_right = jnp.abs(np1_maxs - np2_mins)
            n_left_right = jnp.stack([n_left, n_right], axis=1)
            n_overlap = jnp.min(n_left_right, axis=1)

            x_left = jnp.abs(xp1_mins - xp2_maxs)
            x_right = jnp.abs(xp1_maxs - xp2_mins)
            x_left_right = jnp.stack([x_left, x_right], axis=1)
            x_overlap = jnp.min(x_left_right, axis=1)
            def n_smaller(naxes, n_overlap, xaxes, x_overlap):
                mtv = naxes[n_overlap.argmin()] * n_overlap.min()
                return mtv
                # import pdb; pdb.set_trace()

            def x_smaller(naxes, n_overlap, xaxes, x_overlap):
                mtv = naxes[x_overlap.argmin()] * x_overlap.min()
                # return mtv
                import pdb; pdb.set_trace()
            res = jax.lax.cond(n_overlap.min() < x_overlap.min(), 
                            n_smaller, x_smaller, 
                            *(naxes, n_overlap, xaxes, x_overlap))
            return res
        def did_not_overlap(naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs):
            return jnp.array([0,0,0])

        res = jax.lax.cond(is_overlap, did_overlap, did_not_overlap,
                     *(naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs))

        import pdb;pdb.set_trace()

        # maxs = jnp.max(jnp.stack([A_proj_maxs, B_proj_maxs], axis=1), axis=1)
        # mins = jnp.min(jnp.stack([A_proj_mins, B_proj_mins], axis=1), axis=1)

        # d1 = (A_proj_maxs - A_proj_mins) + (B_proj_maxs - B_proj_mins)
        # d2 = (maxs - mins)
        # overlay = (d1 > d2)
        # assert (overlap==overlay).all()
        return is_overlap, res

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
    