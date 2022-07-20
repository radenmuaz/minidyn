from dataclasses import dataclass
import jax
from jax import numpy as jnp, random
import minidyn.dyn.functions as F
from jax.tree_util import register_pytree_node_class
from functools import partial
import trimesh
import minidyn as mdn
from jax import vmap, tree_map,lax

class Solver:
    pass

class SATSolver(Solver):

    def __init__(self):
        pass

    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, world, qs):
        q1, q2, s1, s2 = [], [], [], []
        for ((ib1, ib2), shape_spairs)  in zip(world.body_pairs_idxs, world.shape_pairs):
            for bs1, bs2 in shape_spairs:
                q1 += qs[ib1][jnp.newaxis,:]
                q2 += qs[ib2][jnp.newaxis,:]
                s1 += [bs1]
                s2 += [bs2]
        q1 = jnp.stack(q1)
        q2 = jnp.stack(q2)
        def stack_attr(pytrees, attr):
            return  tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[getattr(t, attr) for t in pytrees])

        v1 = stack_attr(s1, 'vertices')
        v2 = stack_attr(s2, 'vertices')
        n1 = stack_attr(s1, 'face_normals')
        n2 = stack_attr(s2, 'face_normals')
        f1 = stack_attr(s1, 'faces')
        f2 = stack_attr(s2, 'faces')

        (collide_flags, mtvs, nrefs, p_refs, p_ins) = \
          jax.vmap(self.solve)(q1, q2, v1, v2, n1, n2, f1, f2)
        for var in (collide_flags, mtvs, nrefs, p_refs, p_ins):
            var = jnp.stack(var)
        return (collide_flags, mtvs, nrefs, p_refs, p_ins)

    
    def closest_point(self, p, ve, eps=1e-9):
        result = jnp.zeros(3)[jnp.newaxis, :]
        p = p[jnp.newaxis, :]
        fv = ve[jnp.newaxis, :, :]
        a = fv[:, 0, :]
        b = fv[:, 1, :]
        c = fv[:, 2, :]

        # check if P is in vertex region outside A
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = (ab * ap).sum(axis=1)
        d2 = (ac * ap).sum(axis=1)
        is_a = jnp.logical_and(d1 < eps, d2 < eps)
        result = jnp.where(is_a, a, result)

        # check if P in vertex region outside B
        bp = p - b
        d3 = (ab * bp).sum(axis=1)
        d4 = (ac * bp).sum(axis=1)
        is_b = (d3 > eps) & (d4 <= d3)
        result = jnp.where(is_b, b , result)

        # check if P in edge region of AB, if so return projection of P onto A
        vc = (d1 * d4) - (d3 * d2)
        is_ab = ((vc < eps) &
                (d1 > -eps) &
                (d3 < eps))
        v = (d1 / (d1 - d3)).reshape((-1, 1))
        result = jnp.where(is_ab, a + (v * ab) , result)

        # check if P in vertex region outside C
        cp = p - c
        d5 = (ab * cp).sum(axis=1)
        d6 = (ac * cp).sum(axis=1)
        is_c = (d6 > -eps) & (d5 <= d6)
        result = jnp.where(is_c, c , result)

        # check if P in edge region of AC, if so return projection of P onto AC
        vb = (d5 * d2) - (d1 * d6)
        is_ac = (vb < eps) & (d2 > -eps) & (d6 < eps)
        w = (d2 / (d2 - d6)).reshape((-1, 1))
        result = jnp.where(is_ac, a + w * ac , result)

        # check if P in edge region of BC, if so return projection of P onto BC
        va = (d3 * d6) - (d5 * d4)
        is_bc = ((va < eps) &
             ((d4 - d3) > - eps) &
             ((d5 - d6) > -eps))
        d43 = d4 - d3
        w = (d43 / (d43 + (d5 - d6))).reshape((-1, 1))
        result = jnp.where(is_bc,  b + w * (c - b) , result)

        is_inside = ~(is_a | is_b | is_ab | is_c | is_ac | is_bc)
        denom = 1.0 / (va + vb + vc)
        v = (vb * denom).reshape((-1, 1))
        w = (vc * denom).reshape((-1, 1))
        # compute Q through its barycentric coordinates
        result = jnp.where(is_inside, a + (ab * v) + (ac * w), result)

        return result.squeeze()

  
    def solve(self, q1, q2, v1, v2, n1, n2, f1, f2):
        
        v1 = F.vec2world(v1, F.q2tf(q1)) 
        v2 = F.vec2world(v2, F.q2tf(q2)) 
        
        # for normals, zero translate
        n1 = F.vec2world(n1, F.q2tf(jnp.array([*q1[:4], 0, 0, 0])))
        n2 = F.vec2world(n2, F.q2tf(jnp.array([*q2[:4], 0, 0, 0])))

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
        def did_overlap(v1, v2, n1, n2, f1, f2, naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs):
            n_left = jnp.abs(np1_mins - np2_maxs)
            n_right = jnp.abs(np1_maxs - np2_mins)
            n_left_right = jnp.stack([n_left, n_right], axis=1)
            n_overlap = jnp.min(n_left_right, axis=1)

            x_left = jnp.abs(xp1_mins - xp2_maxs)
            x_right = jnp.abs(xp1_maxs - xp2_mins)
            x_left_right = jnp.stack([x_left, x_right], axis=1)
            x_overlap = jnp.min(x_left_right, axis=1)
            def n_smaller(v1, v2, n1, n2, f1, f2, naxes, n_overlap, xaxes, x_overlap):
                # face-to-face contacts
                length = n_overlap.min()
                i_overlap = n_overlap.argmin()
                Nn1 = len(n1)
                # naxes was [n1, n2], i offset by on i_overlap number
                i_ref = jax.lax.cond(i_overlap < Nn1, lambda i: i, lambda i: i-Nn1, i_overlap)
                n_ref = jax.lax.cond(i_overlap < Nn1, lambda i: n1[i], lambda i: n2[i], i_ref)
                v_ref = jax.lax.cond(i_overlap < Nn1, lambda i: v1[f1[i]], lambda i: v2[f2[i]], i_ref)
                ns = jax.lax.cond(i_overlap < Nn1, lambda x: n2, lambda x: n1, None)
                vs = jax.lax.cond(i_overlap < Nn1, lambda x: v2, lambda x: v1, None)
                fs = jax.lax.cond(i_overlap < Nn1, lambda x: f2, lambda x: f1, None)
                cosine_sim = (jnp.broadcast_to(n_ref, (len(ns),1,3)) @ ns[:,:,jnp.newaxis]).squeeze()
                i_in = jnp.argmin(cosine_sim)
                n_in = ns[i_in]
                v_in = vs[fs[i_in]]
                p0 = self.closest_point(v_ref[0], v_in)
                p_ref = self.closest_point(p0, v_ref)
                p_in = self.closest_point(p_ref, v_in)
                d2 = jnp.dot(p_ref-p_in, n_ref)
                l2 = jnp.linalg.norm(p_ref-p_in)
                
                mtv = n_ref * length
                # print(mtv, length)
                # print((p_ref-p_in), d2)
                # import pdb; pdb.set_trace()
                return jnp.array([True]), mtv, n_ref, p_ref, p_in

            def x_smaller(v1, v2, n1, n2, f1, f2, naxes, n_overlap, xaxes, x_overlap):
                # edge-to-edge contacts
                i_ref = x_overlap.argmin()
                n_ref = naxes[i_ref]
                length = x_overlap.min()
                mtv = n_ref * length
                return jnp.array([True]), mtv, jnp.zeros(3), jnp.zeros(3), jnp.zeros(3)
                # import pdb; pdb.set_trace()
            res = jax.lax.cond(n_overlap.min() < x_overlap.min(), 
                            n_smaller, x_smaller, 
                            *(v1, v2, n1, n2, f1, f2, naxes, n_overlap, xaxes, x_overlap))
            return res
        def did_not_overlap(v1, v2, n1, n2, f1, f2, naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs):
            return jnp.array([False]), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3)

        res = jax.lax.cond(is_overlap, did_overlap, did_not_overlap,
                     *(v1, v2, n1, n2, f1, f2, naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs))

        # import pdb;pdb.set_trace()

        # maxs = jnp.max(jnp.stack([A_proj_maxs, B_proj_maxs], axis=1), axis=1)
        # mins = jnp.min(jnp.stack([A_proj_mins, B_proj_mins], axis=1), axis=1)

        # d1 = (A_proj_maxs - A_proj_mins) + (B_proj_maxs - B_proj_mins)
        # d2 = (maxs - mins)
        # overlay = (d1 > d2)
        # assert (overlap==overlay).all()
        return res

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
    