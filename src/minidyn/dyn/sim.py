import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
import minidyn as mdn
from minidyn.dyn.body import Body
import minidyn.dyn.functions as F
from minidyn.dyn.spatial import Inertia
from minidyn.dyn import integrators 
from minidyn import col
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
import math
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
                    body_pairs=[], shape_pairs=[],
                    static_masks=[]):
        self.root = root
        self.joints = joints
        self.bodies = bodies
        self.shapes = shapes
        self.gravity = gravity
        self.init_qs = init_qs
        self.init_qds = init_qds

        self.body_pairs_idxs = body_pairs_idxs
        self.shape_pairs_idxs = shape_pairs_idxs
        self.body_pairs = body_pairs
        self.shape_pairs = shape_pairs
        self.static_masks = static_masks
    
    def add_ground(self):
        body = Body()
        mass = 1
        moment = jnp.eye(3) * mass
        body.inertia = mdn.dyn.Inertia(mass=mass,moment=moment)
        shape = trimesh.creation.box((100., 100, .1))
        # body.add_shape(mdn.col.Shape.from_trimesh(shape))
        body.shapes = [mdn.col.Shape.from_trimesh(shape)]
        self.add_body(body, static=True,q=jnp.array([1., 0.0, 0, 0., 0, 0. , 0.]))

    def add_body(self, body, q=None, qd=None, static=False):
        q = q if q is not None else jnp.zeros(7).at[0].set(1)
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
            self.body_pairs += [[body, self.bodies[i]]]
            self.shape_pairs += [[[j, k] for j in body.shapes for k in b.shapes]]

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
                    self.body_pairs,
                    self.shape_pairs,
                    self.static_masks
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    

@register_pytree_node_class
class WorldSolver(object):
    def __init__(self, dt=1/30.):
        super().__init__()
        self.integrator = integrators.euler
        self.dt = dt
        self.colsolver = col.solvers.SATSolver()
    
    def __call__(self, world: World, qs, qds, u):
        qdds = self.solve(world, qs, qds, u)
        N = qs.size
        q_vec = cat((qs.reshape(N), qds.reshape(N)))
        qd_vec = cat((qds.reshape(N), qdds.reshape(N)))
        q_vec_new = self.integrator(q_vec, qd_vec, self.dt)
        qs_new = q_vec_new[:N].reshape(qs.shape)
        qds_new  = q_vec_new[N:].reshape(qds.shape)
        return qs_new, qds_new

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, world, qs, qds, u):
        def get_energies(qs, qds, u, world):
            tfs = vmap(F.q2tf)(qs)
            Is_local = tree_map( lambda *values: 
                jnp.stack(values, axis=0), *[body.inertia for body in world.bodies])

            Is = vmap(F.inertia_to_world)(Is_local, tfs)
            vs = vmap(F.qqd2v)(qs, qds)
            Ts = vmap(F.kinetic_energy)(Is, vs)
            g = world.gravity.reshape(1,3)
            # no_g = jnp.zeros(3).reshape(1,3)
            mask = jnp.array(world.static_masks).tile((3,1)).T
            gs = jnp.tile(g.reshape(1,3), len(tfs)).reshape(len(tfs), 3)
            # gs = jnp.tile(world.gravity.reshape(1,3), len(tfs))
            gs = jnp.where(mask == True, 0, gs)
            # gs = gs * jnp.invert(mask).astype(gs.dtype)
            # print(gs.shape)
            Vs = vmap(F.potential_energy)(Is, tfs, gs)
            # import pdb;pdb.set_trace()
            return Ts, Vs
        def L(qs, qds, u, world):
            Ts, Vs = get_energies(qs, qds, u, world)
            # Ts, Vs = self.get_energies(qs, qds, u, world)
            return jnp.sum(Ts) - jnp.sum(Vs)
        N = qs.size
        M = jax.hessian(L, 1)(qs, qds, u, world).reshape(N, N)
        Minv = jnp.linalg.pinv(M, rcond=1e-20)
        g = jax.grad(L, 0)(qs, qds, u, world).reshape(N, 1)
        C =  jax.jacfwd(jax.grad(L, 1), 0)(qs, qds, u, world).reshape(N, N)
        
        colres = self.colsolver(world, qs)
        import pdb;pdb.set_trace()
        
        # C_pen = 

        qd_vec = qds.reshape(N, 1)
        qdds_vec = Minv @ ( g - C @ qd_vec)
        qdds = qdds_vec.reshape(qds.shape)
        # mask = jnp.array(world.static_masks).tile((7,1)).T

        jnp.set_printoptions(precision=4)
        # if colres[0][0][1]==True:
        #     v1=world.bodies[0].shapes[0].vertices
        #     v2=world.bodies[1].shapes[0].vertices
        #     v2w = F.vec2world(v2, F.q2tf(qs[1]))
        #     import pdb;pdb.set_trace()
        print(colres)
        print(world.body_pairs_idxs)
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

class Simulator:
    "View the world"
    def __init__(self, world, world_solver, viz_data=None):
        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(70, 16 / 9)
        self.camera.position = Vector3(10,10,3)
        self.world = world
        self.world_solver = world_solver
        self.qs, self.qds = world.get_init_state()

        self.controls = gfx.OrbitControls(
            eye=self.camera.position.clone(),
                                    target=Vector3(0,0,0),
                                    up=Vector3(0, 0, 1))
        self.controls.add_default_event_handlers(self.renderer, self.canvas, self.camera)
        # self.controls.update_camera(self.camera)

        self.viz_data = viz_data
        if self.viz_data is None:
            body2shapes = []
            for body in self.world.bodies:
                shapes = []
                for shape in body.shapes:
                    trimesh_mesh = trimesh.Trimesh(vertices=shape.vertices,
                        faces=shape.faces)
                    mesh = gfx.Mesh(
                        gfx.trimesh_geometry(trimesh_mesh),
                        gfx.MeshPhongMaterial(),
                    )
                    shapes += [mesh]
                body2shapes += [shapes]
            self.viz_data = {'body2shapes':body2shapes}
        for shapes in self.viz_data['body2shapes']:
            for mesh in shapes:
                self.scene.add(mesh)
        
        
    def start(self):
        self.canvas.request_draw(self.update)
        run()


    def update(self):
        t0 = time.time()
        self.u = jnp.vstack([jnp.zeros(len(qd)) for qd in self.qds])
        self.qs, self.qds = self.world_solver(self.world, self.qs, self.qds, self.u)
        for i in range(len(self.qs)):
            q = self.qs[i]
            shapes = self.viz_data['body2shapes'][i]
            quat = Quaternion(q[0], q[1], q[2], q[3])
            vec3 = Vector3(q[4], q[5], q[6])
            for shape in shapes:
                shape.rotation = quat
                shape.position = vec3
        self.controls.update_camera(self.camera)
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()
        print(f'FPS: {1/(time.time()-t0):.2f}')
        print(self.qs)#, self.qds)

  