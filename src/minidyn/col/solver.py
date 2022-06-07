from dataclasses import dataclass
import jax
from jax import numpy as jnp, random


class Solver:
    pass

class SATSolver(Solver):
    def __init__(self, world):
        self.world = world
        self.body_pairs = []
        for b1 in self.world.bodies:
            for b2 in self.world.bodies:
                if (b1 is not b2) and ([b2, b1] not in self.world.bodies):
                    self.body_pairs += [b1, b2]
        self.shape_pairs = {}
        for i, (b1, b2) in enumerate(self.body_pairs):
            self.shape_pairs[i] = []
            for s1 in b1.shapes:
                for s2 in b2.shapes:
                    if (s1 is not s2) and ([s2, s1] not in self.shape_pairs[i]):
                        self.shape_pairs[i] += [s1, s2]

    def detect(self, qs):
        
        colres = []
        for i in range(len(self.shape_pairs)):
            for (s1, s2) in self.shape_pairs[i]:
                res = self.sat(qs, s1, s2)
            
    def sat(self, qs, s1, s2):
            return
    
            