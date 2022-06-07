import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from pygfx.linalg import Vector3, Matrix4, Quaternion
from minidyn.dyn import functions as F

import minidyn as mdn
from minidyn import dyn
import jax
from jax import numpy as jnp, random
import trimesh

class View:
    "View the world"
    def __init__(self, world):
        self.canvas = WgpuCanvas()
        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(70, 16 / 9)
        self.camera.position.z = 400
        self.world = world
    #     self.cube =  gfx.Mesh(
    #     gfx.box_geometry(100, 200, 200),
    #     gfx.MeshPhongMaterial(color="#336699"),
    # )
        self.body2shapes = []
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
                self.scene.add(mesh)
            self.body2shapes += [shapes]
        self.qs = None

    def animate(self):
        # print('yes')
        # rot = gfx.linalg.Quaternion().set_from_euler(gfx.linalg.Euler(0.005, 0.01))
        # self.cube.rotation.multiply(rot)
        for i in range(len(self.qs)):
            q = self.qs[i]
            shapes = self.body2shapes[i]
            quat = Quaternion(q[0], q[1], q[2], q[3])
            vec3 = Vector3(q[4], q[5], q[6])
            for shape in shapes:
                shape.rotation = quat
                shape.position = vec3
        self.renderer.render(self.scene, self.camera)
        self.canvas.request_draw()

        
if __name__ == "__main__":
    world = dyn.World()

    box_body = dyn.Body()
    box_mass = 1.
    box_moment = jnp.eye(3) * box_mass
    box_body.inertia = dyn.Inertia(mass=box_mass,
                        moment=box_moment,
                        com=jnp.array((0, 0, 0)),
                                        )
    box_shape = trimesh.creation.box((100., 100., 100.))
    box_body.shapes = [mdn.col.Shape.from_trimesh(box_shape)]
    world.add_body(box_body, q=jnp.zeros(7).at[0].set(1))

    box_body2 = dyn.Body()
    box_mass2 = 1.
    box_moment2 = jnp.eye(3) * box_mass
    box_body2.inertia = dyn.Inertia(mass=box_mass2,
                        moment=box_moment2,
                        com=jnp.array((0, 0, 0)),
                                        )
    box_shape2 = trimesh.creation.box((20., 20., 20.))
    box_body2.shapes = [mdn.col.Shape.from_trimesh(box_shape2)]
    world.add_body(box_body2, q=jnp.array([1., 0., 0, 0., 150., 0. , 0.]))

    sim = dyn.Sim()
    qs, qds = world.get_init_state()
    u = jnp.vstack([jnp.zeros(len(qd)) for qd in qds])

    # print('init', qs[0,6], qds[0,6])
    # for i in range(100):
    #     qs, qds = sim(world, qs, qds, u)
    #     print(i, qs, qds)

    view = View(world)

    # ground = gfx.Mesh(
    #     gfx.box_geometry(500, 10, 500),
    #     gfx.MeshPhongMaterial(color="#FFFFFF"),
    # )
    # ground.position = Vector3(0,-100,0)
    # view.scene.add(ground)
 
    # canvas.request_draw(update)
    view.qs = qs
    view.canvas.request_draw(view.animate)

    run()
