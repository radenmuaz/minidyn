from minidyn.dyn import functions as Fn

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
    # box_shape = trimesh.creation.box((1., 1., 1.))
    box_shape = trimesh.creation.capsule(height=1.0, radius=0.5, count=[8,8])
    # breakpoint()
    box_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box_shape)]
    box_body.shapes[0].Kp = 20
    world.add_body(box_body, q=jnp.array([0.892, 0.099, 0.239, 0.370, 0, 0. , 2]), qd=jnp.array([0, 0.05, 0.0, 0., 0, 0. , 0]))

    dt=1/20
    dynamics = mdn.dyn.dynamics.LagrangianDynamics(dt=dt)
    
    sim = mdn.sim.simulator.PygfxSimulator(world, dynamics)
    # dynamics.dt=1/100.

    sim.start()
