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
    ground_body, _, _ = world.add_ground(h=0.2, w=5, q=jnp.array([1., 0.0 , 0, 0., 0, 0. ,1]))
    # world = dyn.World(gravity=jnp.array((0, 0, 0)))

    box1_body = mdn.dyn.body.Body()
    box1_mass = 0.3
    box1_moment = jnp.eye(3) * box1_mass
    box1_body.inertia = mdn.dyn.body.Inertia(mass=box1_mass,
                        moment=box1_moment, com=jnp.array((0, 0, 0)))
    box1_shape = trimesh.creation.box((1, 1, 1))
    box1_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box1_shape)]
    world.add_body(box1_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 0]))

    box2_body = mdn.dyn.body.Body()
    box2_mass = 0.3
    box2_moment = jnp.eye(3) * box1_mass
    box2_body.inertia = mdn.dyn.body.Inertia(mass=box2_mass, moment=box1_moment,
                                com=jnp.array((0, 0, 0)))
    box2_shape = trimesh.creation.box((1., 1., 1.))
    box2_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box2_shape)]
    world.add_body(box2_body, q=jnp.array([1., 0, 0, 0., 0, 1 , -1]))

    a1, a2 = jnp.array([0.0, 0, -2 ]), jnp.array([0., 0, 0 ])
    joint = mdn.dyn.FixedJoint()
    joint.connect(world, ground_body, box1_body, a1, a2)
    world.add_joint(joint)



    dt=1/20
    dynamics = mdn.dyn.dynamics.LagrangianDynamics(dt=dt)
    
    sim = mdn.sim.simulator.PygfxSimulator(world, dynamics)
    # dynamics.dt=1/100.

    sim.start()
