from ._bgo import *
from ._karhunen import *
import GPy
import GPyOpt
from collections import namedtuple
from itertools import product
import itertools
import numpy as np
import pickle

__all__ = ["Box", "Env"]

class Box(object):
    def __init__(self, low, high, shape):
        self.low   = low
        self.high  = high
        self.shape = shape

class Tuple(object):
    def __init__(self, spaces):
        self.space = []
        while spaces:
            self.space.append(spaces.pop(0))



class Env(SkillfulAgent):

    def __init__(self, lengthscale, n_agents, low = 0.0, high = 3.0, name = "APG-v1", 
                       variance=1.0, num_initial_design=2, max_time=100, d=1, n=1000, 
                       energy=0.9, eps=1.e-4):
        self.name = name
        self._action_space = Box(low, high, shape=(2,1))

        self._observation_space = Box(low, high, shape=(4,1))

        with open('/Users/salarsk/developments/phd/nsf/rl_acq/code/apg/'+str(lengthscale)+'.out', 'rb') as f:
            read_in = pickle.load(f)
        self.bgo_list = []
        for i in range(len(read_in)):
            for j in range(len(read_in[i])):
                self.bgo_list.append(read_in[i][j])

        super().__init__(lengthscale, variance, num_initial_design, 
                         max_time, d, n, energy, eps)

    def set_env(self):
        self.get_params()

    def take_random_action(self):
        return np.random.uniform(size=(1,2))*3.0

    def step(self, r_ind):

        return self.bgo_list[r_ind]

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
    

if __name__ == "__main__":
    env = Env(0.1, 2)
    
    print(env.observation_space.shape)
    env.step(2., 2.5, 0.1)
    

