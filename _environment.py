from _agents import *
from _karhunen import *
import GPy
import GPyOpt
from collections import namedtuple
from itertools import product
import itertools
import numpy as np

__all__ = ["Env"]

class Env(SkillfulAgent):

    def __init__(self, lengthscale, name = "APG-v1", variance=1.0, num_initial_design=2, max_time=1000, d=1, n=1000, energy=0.95, eps=1.e-6):
        self.name = name
        super().__init__(lengthscale, variance, num_initial_design, max_time, d, n, energy, eps)

    def reset(self):
        self.get_params()

    def step(self, *action):
        return self.run(action[0], action[1], action[2])

if __name__ == "__main__":
    env = Env(0.1)
    env.reset()
    print(env.step(3.0, 2.8, .4))

