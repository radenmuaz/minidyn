from dataclasses import dataclass
import jax
from jax import numpy as jnp, random

from jax.tree_util import register_pytree_node_class
@register_pytree_node_class
class Body: 
    def __init__(self, inertia=None, shapes=[]):
        self.inertia = inertia
        self.shapes = shapes
    
    def tree_flatten(self):
        children = (self.inertia, self.shapes)
        aux_data = None
        return (children, aux_data)
    
    def add_shape(self,shape):
        self.shapes += [shape,]

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    
    


@register_pytree_node_class
class Inertia:
    def __init__(self, mass, moment, cross_part=None, com=jnp.array((0, 0, 0))):
        self.mass = mass
        self.moment = moment
        self.cross_part = cross_part if cross_part is not None else mass * com
        self.com = com if com is not None else cross_part / mass

    def tree_flatten(self):
        children = (self.mass, self.moment, self.cross_part, self.com)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    

import minidyn.dyn.functions as Fn

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Shape: 
    def __init__(self, vertices, faces, face_normals,
                    Kp=10, Kd=0.5,
                    mu=0.5, alpha=0.1
                    ):
        self.vertices = vertices
        self.faces = faces
        self.face_normals = face_normals
        self.Kp = Kp
        self.Kd = Kd
        self.mu = mu
        self.alpha = alpha
    
    def tree_flatten(self):
        children = (self.vertices, 
                    self.faces, 
                    self.face_normals, 
                    self.Kp, 
                    self.Kd,
                    self.mu, 
                    self.alpha
        )
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)  

    @classmethod
    def from_trimesh(cls, trimesh):
        return cls(jnp.array(trimesh.vertices), 
        jnp.array(trimesh.faces), 
        jnp.array(trimesh.face_normals))
    