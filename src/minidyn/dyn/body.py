from dataclasses import dataclass
import jax
from jax import numpy as jnp, random
from minidyn.dyn.spatial import *

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
    


