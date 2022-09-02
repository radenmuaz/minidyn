from minidyn.dyn import functions as Fn

import minidyn as mdn
from minidyn import dyn
import jax
from jax import numpy as jnp, random
import trimesh
import time
# from jax.config import config
# config.update('jax_disable_jit', True)
if __name__ == "__main__":
    jnp.set_printoptions(precision=3)

    world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 9.81)))
    # world = mdn.dyn.world.World(gravity=jnp.array((0, 0, 0)))
    world.add_ground()
    # world = dyn.World(gravity=jnp.array((0, 0, 0)))

    N = 1
    for i in range(N):
        box_body = mdn.dyn.body.Body()
        box_mass = 0.3-i*0.1
        box_moment = jnp.eye(3) * box_mass
        box_body.inertia = mdn.dyn.body.Inertia(mass=box_mass,
                            moment=box_moment,
                            com=jnp.array((0, 0, 0)),
                                            )
        box_shape = trimesh.creation.box((1.-0.1*i, 1.-0.1*i, 1.-0.1*i))
        box_body.shapes = [mdn.dyn.body.Shape.from_trimesh(box_shape)]
        box_body.alpha = 0.0
        # box_body.shapes[0].Kp = 5
        # world.add_body(box_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 1+i*1.5]))
        world.add_body(box_body, q=jnp.array([.7, 0.2, 0, 0., 0, 0. , 1+i*1.5]))
        # world.add_body(box_body, q=jnp.array([1., 0, 0, 0., 0, 0. , 1+i*.2]), rigid_contact=False)
    dt=1/100
    # dt=1/30
    dynamics = mdn.dyn.dynamics.LagrangianDynamics(dt=dt)

    # q, qd = world.get_init_state()
    # for i in range(1000):
    #     t0 = time.time()
    #     u = jnp.vstack([jnp.zeros(len(qd)) for qd in qd])
    #     q, qd, aux = dynamics(world, q, qd, u)
    #     print(f'{i}, Sim FPS: {1/(time.time()-t0):.2f}\n')
    
    sim = mdn.sim.simulator.PygfxSimulator(world, dynamics)
    # dynamics.dt=1/100.

    sim.start()
