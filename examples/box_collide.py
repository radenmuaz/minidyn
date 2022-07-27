from minidyn.dyn import functions as F

import minidyn as mdn
from minidyn import dyn
import jax
from jax import numpy as jnp, random
import trimesh
# from jax.config import config
# config.update('jax_disable_jit', True)
if __name__ == "__main__":
    jnp.set_printoptions(precision=3)

    world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 9.81)))
    world.add_ground()
    # world = dyn.World(gravity=jnp.array((0, 0, 0)))

    box_body = mdn.dyn.body.Body()
    box_mass = 0.3
    box_moment = jnp.eye(3) * box_mass
    box_body.inertia = mdn.dyn.body.Inertia(mass=box_mass,
                        moment=box_moment,
                        com=jnp.array((0, 0, 0)),
                                        )
    box_shape = trimesh.creation.box((1., 1., 1.))
    box_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box_shape)]
    box_body.shapes[0].Kp = 10
    # world.add_body(box_body, q=jnp.array([1., 0.0, 0, 0., 0, 0. , 1.5]),qd=jnp.array([1e-9, 0.1, 0, 0., 0, 0. , 0]))
    # world.add_body(box_body, q=jnp.array([1., 0.0, 0, 0., 0, 0. , 1.5]),qd=jnp.array([1e-9, 0.1, 0, 0., 0, 0. , 0]))
    world.add_body(box_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 0.95]))

    # box_body2 = dyn.Body()
    # box_mass2 = 1.
    # box_moment2 = jnp.eye(3) * box_mass
    # box_body2.inertia = dyn.Inertia(mass=box_mass2,
    #                     moment=box_moment2,
    #                     com=jnp.array((0, 0, 0)),
    #                                     )
    # box_shape2 = trimesh.creation.box((0.5, 0.5, 0.5))
    # box_body2.shapes = [mdn.dyn.body.Shape.from_trimesh(box_shape2)]
    # box_body2.shapes[0].Kp = 0.3
    # world.add_body(box_body2, q=jnp.array([1., 0., 0, 0., 0., 1. , 1.]),
    #                         qd=jnp.array([1e-9, 0, 0, 0., 0, 0. , 0]))

    # box_body3 = dyn.Body()
    # box_mass3 = 1.
    # box_moment3 = jnp.eye(3) * box_mass
    # box_body3.inertia = dyn.Inertia(mass=box_mass2,
    #                     moment=box_moment2,
    #                     com=jnp.array((0, 0, 0)),
    #                                     )
    # box_shape3 = trimesh.creation.box((0.5, 0.5, 0.5))
    # box_body3.shapes = [mdn.dyn.body.Shape.from_trimesh(box_shape2)]
    # box_body3.shapes[0].Kp = 0.3
    # world.add_body(box_body3, q=jnp.array([1., 0., 0, 0., 0., -1. , 1.]),
    #                         qd=jnp.array([1e-9, 0, 0, 0., 0, 0. , 0]))

    dynamics = mdn.dyn.dynamics.LagrangianDynamics()
    
    sim = mdn.sim.simulator.PygfxSimulator(world, dynamics)
    sim.start()
