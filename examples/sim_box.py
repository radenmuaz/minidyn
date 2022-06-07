import minidyn as mdn
from minidyn import dyn
import jax
from jax import numpy as jnp, random

world = dyn.World()
box_body = dyn.Body()
box_mass = 1.
box_moment = jnp.eye(3) * box_mass
box_body.inertia = dyn.Inertia(mass=box_mass,
                    moment=box_moment,
                    com=jnp.array((0, 0, 0)),
                                    )


world.add_body(box_body)

sim = dyn.Sim()
qs, qds = world.get_init_state()
u = jnp.vstack([jnp.zeros(len(qd)) for qd in qds])


print('init', qs[0,6], qds[0,6])
for i in range(100):
    qs, qds = sim(world, qs, qds, u)
    print(i, qs, qds)
    # print(i, qs[0,6], qds[0,6])
# def jit_step(world, qs, qds, u):
#     for i in range(100):
#         qs, qds = sim(world, qs, qds, u[i])
#         print(i, qs[0,6], qds[0,6])
# step = jax.jit(jit_step)
# step(world, qs, qds, u)