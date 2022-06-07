from xml.sax.handler import feature_namespaces
import jax
from jax import numpy as jnp, random
from minidyn.dyn.spatial import *

from jax.tree_util import register_pytree_node_class
@register_pytree_node_class
class Shape: 
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
    
    def tree_flatten(self):
        children = (self.vertices, self.faces)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)  

    @classmethod
    def from_trimesh(cls, trimesh):
        # import pdb;pdb.set_trace()
        return cls(jnp.array(trimesh.vertices), jnp.array(trimesh.faces))
    