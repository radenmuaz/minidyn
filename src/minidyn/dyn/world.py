import jax
from jax import numpy as jnp, random
from jax.numpy import concatenate as cat

from minidyn.dyn.body import Body, Inertia, Shape
from minidyn.dyn.contacts import RigidContact
from minidyn.dyn.joints import FixedJoint
from minidyn.dyn.collisions import separating_axis
import minidyn.dyn.functions as Fn

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
from minidyn.dyn import functions as Fn
# functions = [func1, func2, func3]
# index = jnp.arange(len(functions))
# x = jnp.ones((3, 5))

# vmap_functions = vmap(lambda i, x: lax.switch(i, functions, x))
# vmap_functions(index, x)

@register_pytree_node_class
class World: 
    '''
    Bodies connected with joints
    '''
    def __init__(self, joints=[], fixed_joints=[], rigid_contacts=[],
                     bodies=[],gravity=jnp.array((0, 0, 9.81)),
                    init_qs=[], init_vs=[]
                    ):
        self.joints = joints
        self.fixed_joints = fixed_joints
        self.rigid_contacts = rigid_contacts
        self.bodies = bodies
        self.gravity = gravity
        self.init_qs = init_qs
        self.init_vs = init_vs
    
        

    
    def add_ground(self, h=2, w=10, q=None, Kp=1):
        body = Body()
        mass = 0
        moment = jnp.eye(3) * mass
        body.inertia = Inertia(mass=mass,moment=moment)
        shape = trimesh.creation.box((w, w, h))
        body.shapes = [Shape.from_trimesh(shape)]
        body.shapes[0].Kp = Kp
        if q is None:
            q = jnp.array([1., 0.0 , 0, 0., 0, 0. , -h/2])
        self.add_body(body, q=q)
        return body, body.inertia, shape

    def add_body(self, body, q=None, v=None, rigid_contact=False):
        # q = q if q is not None else jnp.zeros(7).at[0].set(1)
        q = jnp.concatenate([Fn.quat_norm(q[:4]), q[4:]]) if q is not None else jnp.zeros(7).at[0].set(1)
        v = v if v is not None else jnp.zeros(6)#.at[0].set(1e-9)
        self.bodies += [body,]
        if rigid_contact:
            for other_body in self.bodies:
                if other_body is body:
                    continue
                for other_shape in other_body.shapes:
                    for shape in body.shapes:
                        rigid_contact = RigidContact()
                        rigid_contact.connect(self, body, other_body, shape, other_shape)
                        self.add_rigid_contact(rigid_contact)

        self.init_qs += [q]
        self.init_vs += [v]
        
    
    def add_joint(self, joint):
        self.joints += [joint,]
        if type(joint) is FixedJoint:
            self.fixed_joints += [joint,]

    
    
    def add_rigid_contact(self, rigid_contact):
        self.rigid_contacts += [rigid_contact,]
    
    def get_init_state(self):
        return jnp.vstack(self.init_qs), jnp.vstack(self.init_vs)
    
    def tree_flatten(self):
        children = (
                    self.joints,
                    self.fixed_joints,
                    self.rigid_contacts,
                    self.bodies,
                    self.gravity,
                    self.init_qs,
                    self.init_vs,
                    )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    
