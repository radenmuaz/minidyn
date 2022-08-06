import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat
# import numpy as jnp
import minidyn as mdn
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
import numpy as np
class PygfxSimulator:
    "View the world"
    def __init__(self, world, dynamics, viz_data=None):
        self.canvas = WgpuCanvas(max_fps=1/dynamics.dt)
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(70, 16 / 9)
        # self.camera.position = Vector3(10,10,3)
        self.camera.position = Vector3(5,5,1)
        self.world = world
        self.dynamics = dynamics
        self.q, self.qd = world.get_init_state()

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
                        # gfx.MeshBasicMaterial(wireframe=True)
                        gfx.MeshPhongMaterial(shininess=np.random.randint(5,20), emissive=np.random.rand(4)-0.3),
                        # gfx.MeshPhongMaterial(emissive=(0, 0, 0, 0)),
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
        self.u = jnp.vstack([jnp.zeros(len(qd)) for qd in self.qd])
        self.q, self.qd, aux = self.dynamics(self.world, self.q, self.qd, self.u)
        for i in range(len(self.q)):
            q = self.q[i]
            shapes = self.viz_data['body2shapes'][i]
            quat = q[:4]
            quat = F.quat_norm(quat)
            quat = Quaternion(quat[1], quat[2], quat[3], quat[0])
            # quat = Quaternion(q[1], q[2], q[3],q[0])
            vec3 = Vector3(q[4], q[5], q[6])
            for shape in shapes:
                shape.rotation = quat
                shape.position = vec3
        self.controls.update_camera(self.camera)
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()
        print(f'Sim FPS: {1/(time.time()-t0):.2f}\n')

        for i, q in enumerate(self.q): print(f'q{i}: ', q)
        print()
        for i, qd in enumerate(self.qd): print(f'qd{i}: ', qd)
        print()
        return
        for i, F_c in enumerate(aux[0]): print(f'F_c{i}: ', F_c)
        print()
        print('collisions:',aux[1][1])
        if aux[1][1].any(): print(aux[1][5:])
        # if aux[1][1].any(): print(aux[1][5:]);breakpoint()
        # if aux[1][2].any(): breakpoint()
        # if aux[1][3].any(): breakpoint()
        # print('Lmult',aux[2])
        # print('J',aux[3].reshape(self.q.shape[0],-1))
        # print('Jd',aux[4].reshape(self.q.shape[0],-1))
        # print('JL',aux[5].reshape(self.qs.shape[0],-1))
  