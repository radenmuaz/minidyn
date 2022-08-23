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

    # world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 9.81)))
    world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 0)))
    # world.add_ground()
    # world = dyn.World(gravity=jnp.array((0, 0, 0)))

    N = 2
    for i in range(N):
        box_body = mdn.dyn.body.Body()
        box_mass = 0.3
        box_moment = jnp.eye(3) * box_mass
        box_body.inertia = mdn.dyn.body.Inertia(mass=box_mass,
                            moment=box_moment,
                            com=jnp.array((0, 0, 0)),
                                            )
        box_shape = trimesh.creation.box((1., 1., 1.))
        box_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box_shape)]
        # box_body.shapes[0].Kp = 5
        # world.add_body(box_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 1+i*.2]))
        world.add_body(box_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 1+i*.2]), rigid_contact=False)
    dt=1/60
    dynamics = mdn.dyn.dynamics.LagrangianDynamics(dt=dt)
    
    sim = mdn.sim.simulator.PygfxSimulator(world, dynamics)
    # dynamics.dt=1/100.

    sim.start()
