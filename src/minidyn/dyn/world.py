import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat

import minidyn as mdn
from minidyn.dyn.body import Body, Inertia
import minidyn.dyn.functions as F

from typing import *
from functools import partial
from jax import vmap, tree_map,lax
from jax.tree_util import register_pytree_node_class
import time
import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.linalg import Vector3, Matrix4, Quaternion
import trimesh
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
                     bodies=[], shapes=[],gravity=jnp.array((0, 0, 9.81)),
                    init_qs=[], init_qds=[],
                    body_pairs_idxs=[], shape_pairs_idxs=[], 
                    body_pairs_mat_idxs=[], shape_pairs_mat_idxs=[],
                    body_pairs=[], shape_pairs=[],
                    body_pairs_mat=[], shape_pairs_mat=[],
                    static_masks=[]):
        self.root = root
        self.joints = joints
        self.bodies = bodies
        self.shapes = shapes
        self.gravity = gravity
        self.init_qs = init_qs
        self.init_qds = init_qds
    
        self.body_pairs = body_pairs
        self.shape_pairs = shape_pairs
        self.body_pairs_mat = body_pairs_mat # body pairs tiled to match shapes len
        self.shape_pairs_mat = shape_pairs_mat # body pairs tiled to match shapes len

        self.body_pairs_idxs = body_pairs_idxs
        self.shape_pairs_idxs = shape_pairs_idxs
        self.body_pairs_mat_idxs = body_pairs_mat_idxs 
        self.shape_pairs_mat_idxs = shape_pairs_mat_idxs 
        

        self.static_masks = static_masks
    
    def add_ground(self):
        body = Body()
        mass = 1e-5
        moment = jnp.eye(3) * mass
        body.inertia = mdn.dyn.body.Inertia(mass=mass,moment=moment)
        # shape = trimesh.creation.box((100., 100, .1))
        h = 100
        w = 100
        shape = trimesh.creation.box((w, w, h))
        # body.add_shape(mdn.col.Shape.from_trimesh(shape))
        body.shapes = [mdn.dyn.body.Shape.from_trimesh(shape)]
        self.add_body(body, static=True,q=jnp.array([1., 0.0, 0, 0., 0, 0. , -h/2]))

    def add_body(self, body, q=None, qd=None, static=False):
        q = q if q is not None else jnp.zeros(7).at[0].set(1)
        # qd = qd if qd is not None else jnp.zeros(7).at[0].set(1e-18)
        qd = qd if qd is not None else jnp.zeros(7).at[0].set(1e-9)
        self.init_qs += [q]
        # self.init_qs += [jnp.zeros(7).at[0].set(1).at[6].set(10)]
        self.init_qds += [qd]
        # self.init_qds += [jnp.zeros(7).at[0].set(1e-9).at[1].set(1)]
        # self.graph.add_node(body)
        def rl(x): return range(len(x))
        N = len(self.bodies)
        for i, b in enumerate(self.bodies):
            self.body_pairs_idxs += [[N, i]]
            self.shape_pairs_idxs += [[[j, k] for j in rl(body.shapes) for k in rl(b.shapes)]]
            self.body_pairs_mat_idxs += [[N, i]*(len(b.shapes)*len(body.shapes))]
            self.shape_pairs_mat_idxs += [[j, k] for j in rl(body.shapes) for k in rl(b.shapes)]
            self.body_pairs += [[body, self.bodies[i]]]
            self.shape_pairs += [[[j, k] for j in body.shapes for k in b.shapes]]
            self.body_pairs_mat += [[body, self.bodies[i]]*(len(b.shapes)*len(body.shapes))]
            self.shape_pairs_mat += [[j, k] for j in body.shapes for k in b.shapes]

        self.bodies += [body,]
        self.static_masks += [static]
        
        
    
    def add_joint(self, joint, body, pred_body):
        self.joints += [joint,]
    
    def get_init_state(self):
        return jnp.vstack(self.init_qs), jnp.vstack(self.init_qds)
    
    def tree_flatten(self):
        children = (
                    self.root,
                    self.joints,
                    self.bodies,
                    self.shapes,
                    self.gravity,
                    self.init_qs,
                    self.init_qds,
                    self.body_pairs_idxs,
                    self.shape_pairs_idxs,
                    self.body_pairs_mat_idxs,
                    self.shape_pairs_mat_idxs,
                    self.body_pairs,
                    self.shape_pairs,
                    self.body_pairs_mat,
                    self.shape_pairs_mat,
                    self.static_masks
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    
