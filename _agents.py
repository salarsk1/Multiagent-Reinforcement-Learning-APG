''' This code evaluates the conergence of the 
    Bayesian Global Optimization
'''
__all__ = ['SkillfulAgent']

from mpi4py import MPI
import pickle
import numpy as np
import GPy
import GPyOpt
import sys
from _karhunen import *
from collections import namedtuple
from itertools import product
import itertools
import matplotlib.pyplot as plt


class EmbarrassParallel(object):
    '''
    Define the methods to run the search in parallel
    num_samples: Number of random function
    size: the number of the cpu's
    '''

    def __init__(self, n_samples, size):
        
        self.n_samples = n_samples
        self.size      = size

    def split(self):

        jobs = list(range(self.n_samples))
        return [jobs[i::self.size] for i in range(self.size)]



class SkillfulAgent(object):

    '''
    Evaluates how a skillful agent improves the quality of the search

    lengthscale: Smoothness of the drawn functions
    max_time: maximum time to evaluate the BGO
    max_iter: maximum number of iterations in BGO
    d: the dimension of the functiuons
    n: number of test points
    '''

    def __init__(self, lengthscale, variance, num_initial_design, max_time, d, n, energy, eps):

        self.max_time           = max_time
        self.d                  = d
        self.n                  = n
        self.lengthscale        = lengthscale
        self.variance           = variance
        self.num_initial_design = num_initial_design
        self.energy             = energy
        self.eps                = eps


    def bayes_opt(self, b, q, cost, karhunen, minimum, maximum, xi, indc, domain = (0, 1.0), 
                  type_initial_design = 'random', acquisition = 'EI', 
                  normalize_Y = True, exact_feval = True):

        bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': domain, 
                  'dimensionality':self.d}]
        # xi   = np.random.randn(len(indc))
        myBoptnD = GPyOpt.methods.BayesianOptimization(f=lambda x, ksi=xi, kle = karhunen, minval=minimum, maxval = maximum, ic=indc : 
                                                       -self.f_ndsample(kle, ksi, x, minval, maxval, ic),
                                                       domain=bounds,
                                                       initial_design_numdata = self.num_initial_design,
                                                       initial_design_type=type_initial_design,
                                                       acquisition_type=acquisition,
                                                       normalize_Y = normalize_Y,
                                                       exact_feval = exact_feval)
        max_iter = int(b / cost)
        myBoptnD.run_optimization(max_iter, self.max_time, eps=self.eps)
        return -myBoptnD.Y_best[2:]

    def get_min_max(self, kle, xi, x, indc):

        sum = 0.0
        phi = np.ndarray(shape=(x.shape[0],len(indc)))
        PHI = []
        for d in range(len(indc[0])):
            PHI.append(kle.eval_phi(x[:,d].flatten()[:,None]))
        count = 0
        PHI = np.array(PHI[0])
        dim = len(indc[0])
        for tt in range(len(indc)):
            ic = indc[tt]
            temp = 1.0
            for d in range(dim):
                temp *= PHI[:,ic[d]]*kle.sqrt_lam[ic[d]]
            phi[:,count] = temp
            count += 1
        f = np.dot(phi, xi)
        return min(f), max(f)

    def f_ndsample(self, kle, xi, x, minimum, maximum, indc):

        sum = 0.0
        phi = np.ndarray(shape=(x.shape[0],len(indc)))
        PHI = []
        for d in range(len(indc[0])):
            PHI.append(kle.eval_phi(x[:,d].flatten()[:,None]))
        count = 0
        PHI = np.array(PHI[0])
        dim = len(indc[0])
        for tt in range(len(indc)):
            ic = indc[tt]
            temp = 1.0
            for d in range(dim):
                temp *= PHI[:,ic[d]]*kle.sqrt_lam[ic[d]]
            phi[:,count] = temp
            count += 1
        f = np.dot(phi, xi)
        avg = 0.5 * (maximum + minimum)
        div = 0.5 * (maximum - minimum)
        f -= avg
        f /= div
        f *= 3.0
        return f

    def get_params(self):
        k = GPy.kern.RBF(1,variance = self.variance, lengthscale=self.lengthscale)
        self.kle  = KarhunenLoeveExpansion(k, nq=self.n, alpha = self.energy)
        self.indc = list(product(range(self.kle.num_xi), repeat=self.d))
        self.xi = np.random.randn(len(self.indc))
        self.minimum, self.maximum = self.get_min_max(self.kle, self.xi, np.linspace(0, 1.0, 1000).reshape(-1,1), self.indc)


    def run(self, b, q, cost):
        return self.bayes_opt(b, q, cost, self.kle, self.minimum, self.maximum, self.xi, self.indc)


if __name__ == "__main__":
    b = 3.0
    q = 2.5
    cost = 0.1
    agent = SkillfulAgent(0.01, 1.0, 2, 2000, 1, 1000, 0.95, 1.0e-6)
    agent.get_params()
    result = agent.run(b, q, cost)
    print(result)

