# from __future__ import annotations

from dataclasses import dataclass
import jax
from jax import numpy as jnp, random
from typing import *
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Inertia:
    def __init__(self, mass, moment, cross_part=None, com=None, frame=None):
        self.mass = mass
        self.moment = moment
        self.cross_part = cross_part if cross_part is not None else mass * com
        self.com = com if com is not None else cross_part / mass
        self.frame = frame

    def tree_flatten(self):
        children = (self.mass, self.moment, self.cross_part, self.com, self.frame)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)    

