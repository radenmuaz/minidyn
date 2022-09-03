from minidyn.dyn import functions as Fn

import minidyn as mdn
from minidyn import dyn
import jax
from jax import numpy as jnp, random
import trimesh
# from jax.config import config
# config.update('jax_disable_jit', True)
if __name__ == "__main__":
    jnp.set_printoptions(precision=2)

    # world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 9.81)))
    world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 0)))
    # ground_body, _, _ = world.add_ground(h=0.2, w=5, q=jnp.array([1., 0.0 , 0, 0., 0, 0. ,1]))
    # world = dyn.World(gravity=jnp.array((0, 0, 0)))

    world_body = mdn.dyn.body.Body()
    world_mass = 0
    world_moment = jnp.eye(3) * world_mass
    world_body.inertia = mdn.dyn.body.Inertia(mass=world_mass,
                        moment=world_moment, com=jnp.array((0, 0, 0)))
    world_shape = trimesh.creation.box((.1, .1, .1))
    world_body.shapes = [mdn.dyn.body.Shape.from_trimesh(world_shape)]
    world.add_body(world_body, q=jnp.array([1., 0, 0, 0., 0, 0 , 0]))

    box1_body = mdn.dyn.body.Body()
    box1_mass = 100
    box1_moment = jnp.eye(3) * box1_mass
    box1_body.inertia = mdn.dyn.body.Inertia(mass=box1_mass,
                        moment=box1_moment, com=jnp.array((0, 0, 0)))
    box1_shape = trimesh.creation.box((5, 5, 1))
    box1_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box1_shape)]
    world.add_body(box1_body, q=jnp.array([.9, 0.1, 0, 0. ,0, 0 , 0]))
    # world.add_body(box1_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 0]), qd=jnp.array([0.1, 0.1, 0, 0., 0, 0. , 0]))

    

    a1, a2 = jnp.array([0.0, 0, 0 ]), jnp.array([0., 0, 0])
    joint = mdn.dyn.FixedJoint()
    joint.connect(world, box1_body, world_body, a1, a2)
    world.add_joint(joint)

    # box2_body = mdn.dyn.body.Body()
    # box2_mass = 1
    # box2_moment = jnp.eye(3) * box1_mass
    # box2_body.inertia = mdn.dyn.body.Inertia(mass=box2_mass, moment=box1_moment,
    #                             com=jnp.array((0, 0, 0)))
    # box2_shape = trimesh.creation.box((1., 1., 1.))
    # box2_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box2_shape)]
    # world.add_body(box2_body, q=jnp.array([1., 0, 0, 0., 0, 1 , 1.2]))
    # rigid_contact = mdn.dyn.RigidContact()
    # rigid_contact.connect(world, box1_body, box2_body, box1_body.shapes[0], box2_body.shapes[0])
    # world.add_rigid_contact(rigid_contact)



    dt=1/20
    dynamics = mdn.dyn.dynamics.LagrangianDynamics(dt=dt)
    
    sim = mdn.sim.simulator.PygfxSimulator(world, dynamics)
    # dynamics.dt=1/100.

    sim.start()
