from minidyn.dyn import functions as F

import minidyn as mdn
from minidyn import dyn
import jax
from jax import numpy as jnp, random
import trimesh
      
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

    world_solver = dyn.WorldSolver()
    
    sim = dyn.Simulator(world, world_solver)
    sim.start()
    # run()
    # for i in range(100):
    #     qs, qds = sim(world, qs, qds, u)
    #     import pdb;pdb.set_trace()
