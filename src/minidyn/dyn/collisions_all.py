from dataclasses import dataclass
import jax
from jax import numpy as jnp, random
import minidyn.dyn.functions as Fn
from jax.tree_util import register_pytree_node_class
from functools import partial
import trimesh
import minidyn as mdn
from jax.tree_util import tree_map



@register_pytree_node_class
class SeparatingAxis:

    def __init__(self,body_pairs_idxs=[], shape_pairs_idxs=[], 
                    body_pairs_mat_idxs=[], shape_pairs_mat_idxs=[],
                    body_pairs=[], shape_pairs=[],
                    body_pairs_mat=[], shape_pairs_mat=[]):
        self.body_pairs = body_pairs
        self.shape_pairs = shape_pairs
        self.body_pairs_mat = body_pairs_mat # body pairs tiled to match shapes len
        self.shape_pairs_mat = shape_pairs_mat # body pairs tiled to match shapes len
        self.body_pairs_idxs = body_pairs_idxs
        self.shape_pairs_idxs = shape_pairs_idxs
        self.body_pairs_mat_idxs = body_pairs_mat_idxs 
        self.shape_pairs_mat_idxs = shape_pairs_mat_idxs 
    
    def add_body(self, world, body):
        def rl(x): return range(len(x))
        N = len(world.bodies)
        for i, b in enumerate(world.bodies):
            self.body_pairs_idxs += [[N, i]]
            self.shape_pairs_idxs += [[[j, k] for j in rl(body.shapes) for k in rl(b.shapes)]]
            self.body_pairs_mat_idxs += [[N, i]*(len(b.shapes)*len(body.shapes))]
            self.shape_pairs_mat_idxs += [[j, k] for j in rl(body.shapes) for k in rl(b.shapes)]
            self.body_pairs += [[body, world.bodies[i]]]
            self.shape_pairs += [[[j, k] for j in body.shapes for k in b.shapes]]
            self.body_pairs_mat += [[body, world.bodies[i]]*(len(b.shapes)*len(body.shapes))]
            self.shape_pairs_mat += [[j, k] for j in body.shapes for k in b.shapes]

    
    # @partial(jax.jit, static_argnums=(0,))
    def __call__(self, q):
        q1, q2, s1, s2 = [], [], [], []
        for (ib1, ib2), (bs1, bs2) in zip(self.body_pairs_mat_idxs, self.shape_pairs_mat):
            q1 += [q[ib1]]#[jnp.newaxis,:]]
            q2 += [q[ib2]]#[jnp.newaxis,:]]
            s1 += [bs1]
            s2 += [bs2]
        # import pdb;pdb.set_trace()
        q1 = jnp.stack(q1)
        q2 = jnp.stack(q2)
        def Fn.stack_attr(pytrees, attr):
            return  tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[getattr(t, attr) for t in pytrees])

        v1 = Fn.stack_attr(s1, 'vertices')
        v2 = Fn.stack_attr(s2, 'vertices')
        n1 = Fn.stack_attr(s1, 'face_normals')
        n2 = Fn.stack_attr(s2, 'face_normals')
        f1 = Fn.stack_attr(s1, 'faces')
        f2 = Fn.stack_attr(s2, 'faces')
        # import pdb; pdb.set_trace()

        # with jax.disable_jit():
        (did_collide, face2face, other, mtvs, nrefs, p_refs, p_ins) = \
          jax.vmap(self.solve)(q1, q2, v1, v2, n1, n2, f1, f2)
        for var in (did_collide, face2face, other, mtvs, nrefs, p_refs, p_ins):
            var = jnp.stack(var)
        # import pdb;pdb.set_trace()
        
        return (did_collide, face2face, other, mtvs, nrefs, p_refs, p_ins)

    
    
    def solve(self, q1, q2, v1, v2, n1, n2, f1, f2):
        
        v1 = Fn.vec2world(v1, Fn.q2tf(q1)) 
        v2 = Fn.vec2world(v2, Fn.q2tf(q2)) 
        
        # for normals, zero translate
        n1 = Fn.vec2world(n1, Fn.q2tf(jnp.array([*q1[:4], 0, 0, 0])))
        n2 = Fn.vec2world(n2, Fn.q2tf(jnp.array([*q2[:4], 0, 0, 0])))

        naxes = jnp.concatenate([n1, n2], axis=0)
        def build_edge_vec(v):
            return Fn.vec_normalize(v - jnp.roll(v,-1,0))
            # return v - jnp.roll(v,-1,0)
        e1 = build_edge_vec(v1)
        e2 = build_edge_vec(v2)
        xaxes = jnp.reshape(jnp.cross(e1[:, jnp.newaxis, :], e2), (-1, 3))
        idxzeros = jnp.all(xaxes== 0, axis=1).reshape(-1, 1)
        replace = jnp.tile(jnp.array([1,0,0]), (len(xaxes),1))
        # replace = jnp.array([1,0,0]).tile((len(xaxes),1))
        # import pdb;pdb.set_trace()
        xaxes = jnp.where(idxzeros, replace, xaxes)
        # idxzeros = jnp.broadcast_to(idxzeros, (idxzeros.shape[0], 3))
        # @partial(jax.jit, static_argnums=(0,))

        # xaxes[idxzeros,:] = jnp.array([1.,0,0])
        # xaxes = xaxes[~jnp.all(xaxes== 0, axis=1)] # prune zero vectors
        xaxes = Fn.vec_normalize(xaxes)
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
            n1_diff = np1_maxs - np2_mins
            n2_diff = np2_maxs - np1_mins
            n1_overlap = n1_difFn.min()
            n2_overlap = n2_difFn.min()
            i1 = n1_difFn.argmin()
            i1 = jnp.where(i1>len(n1),i1-len(n2),i1)
            i2 = n2_difFn.argmin()
            i2 = jnp.where(i2>len(n2),i2-len(n1),i2)

            x1_diff = xp1_maxs - xp2_mins
            x2_diff = xp2_maxs - xp1_mins
            x1_overlap = jnp.min(x1_diff)
            x2_overlap = jnp.min(x2_diff)
            # breakpoint()

            def n_smaller(v1, v2, n1, n2, f1, f2, i, n_overlap, n_diff):
                n_ref = n1[i]
                v_ref = v1[f1[i]]
                cosine_sim = (jnp.broadcast_to(n_ref, (len(n2),1,3)) @ n2[:,:,jnp.newaxis]).squeeze()
                i_in = jnp.argmin(cosine_sim)
                # n_in = n2[i_in]
                v_in = v2[f2[i_in]]
                p0 = self.closest_point(v_ref[0], v_in)
                p_ref = self.closest_point(p0, v_ref)
                p_in = self.closest_point(p_ref, v_in)
                # p_ref = self.closest_point(p_in, v_ref)
                # d2 = jnp.dot(p_ref-p_in, n_ref)
                # l2 = jnp.linalg.norm(p_ref-p_in)
                length = n_overlap
                mtv = n_ref * length
                # breakpoint()
                return jnp.stack((jnp.array([True,True,False]), mtv, n_ref, p_ref, p_in))

            def x_smaller(v1, v2, n1, n2, f1, f2, i, x_overlap):
                # edge-to-edge contacts
                i_ref = x_overlap.argmin()
                n_ref = naxes[i_ref]
                
                e1s, e1e = v1[i_ref], v1[i_ref-1]
                e2s, e2e = v2[i_ref], v2[i_ref-1]
                d1 = Fn.vec_normalize((e1e - e1s)[jnp.newaxis,:]).squeeze()
                d2 = Fn.vec_normalize((e2e - e2s)[jnp.newaxis,:]).squeeze()
                d2_perp = jnp.cross(d2, n_ref)
                d2_perp = Fn.vec_normalize((d2_perp)[jnp.newaxis,:]).squeeze()

                m = (jnp.dot(-d2_perp, (e1s-e2s)) / jnp.dot(d2_perp, d1))
                p_ref = e1s + m * d1
                p_in = p_ref

                length = x_overlap.min()
                mtv = n_ref * length
                # import pdb; pdb.set_trace()
                return jnp.stack((jnp.array([True,False,True]), mtv, n_ref, p_ref, p_in))
                # import pdb; pdb.set_trace()

            res = jnp.where(
                    jnp.stack([n1_overlap, n2_overlap]).min() < jnp.stack([x1_overlap, x2_overlap]).min(),
                    # if face2face
                    jnp.where(n1_overlap < n2_overlap,
                        n_smaller(v1, v2, n1, n2, f1, f2, i1, n1_overlap, n1_diff),
                        n_smaller(v2, v1, n2, n1, f2, f1, i2, n2_overlap, n2_diff)),
                    # elif edge2dge
                    jnp.where(x1_overlap < x2_overlap,
                        x_smaller(v1, v2, n1, n2, f1, f2, i1, x1_overlap),
                        x_smaller(v2, v1, n2, n1, f2, f1, i2, x2_overlap))
            )

            return res
        def did_not_overlap():
            return jnp.stack((jnp.array([False]*3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3)))

        res = jnp.where(is_overlap, 
                    did_overlap(v1, v2, n1, n2, f1, f2, naxes, np1_mins, np1_maxs, np2_mins, np2_maxs,
                        xaxes, xp1_mins, xp1_maxs, xp2_mins, xp2_maxs),
                    did_not_overlap()
                     )
                
        (collide_flags, mtvs, nrefs, p_refs, p_ins) = res[0],res[1],res[2],res[3],res[4]
        did_collide = collide_flags[0]
        face2face = collide_flags[1]
        other = collide_flags[2]
        # (collide_flags, mtvs, nrefs, p_refs, p_ins) = jnp.split(res, 5)
        return  (did_collide, face2face, other, mtvs, nrefs, p_refs, p_ins)

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
    
    def tree_flatten(self):
        children = (
                    self.body_pairs_idxs,
                    self.shape_pairs_idxs,
                    self.body_pairs_mat_idxs,
                    self.shape_pairs_mat_idxs,
                    self.body_pairs,
                    self.shape_pairs,
                    self.body_pairs_mat,
                    self.shape_pairs_mat
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    

  
if __name__ == '__main__':
    solver = SeparatingAxis()
    q1 = jnp.array([1, 0, 0, 0., 0, 0, 0]) 
    # q2 = jnp.array([1, 0, 0, 0., 0.99, 0.99, 0.99]) # collide
    # q2 = jnp.array([1, 0, 0, 0., 1.01, 0, 0]) # not collide
    q2 = jnp.array([1, 0, 0, 0., 0, 0, 10]) # not collide
    # s1 = mdn.col.Shape.from_trimesh(trimesh.creation.box((1., 1., 1.)))
    s1 = mdn.dyn.body.Shape.from_trimesh(trimesh.creation.box((10., 0.1, 10.)))
    s2 = mdn.dyn.body.Shape.from_trimesh(trimesh.creation.box((1., 1., 1.)))
    overlap, overlap_all = solver.solve(q1,q2,s1,s2)
    print(overlap)
    print(overlap_all)
    # import pdb;pdb.set_trace()
    
