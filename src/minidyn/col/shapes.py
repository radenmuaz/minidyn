from xml.sax.handler import feature_namespaces
import jax
from jax import numpy as jnp, random
from minidyn.dyn.spatial import *
import minidyn.dyn.functions as F

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Shape: 
    def __init__(self, vertices, faces, face_normals):
        self.vertices = vertices
        self.faces = faces
        self.face_normals = face_normals
    
    def tree_flatten(self):
        children = (self.vertices, 
                    self.faces, 
                    self.face_normals, 
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
    