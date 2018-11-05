import numpy as np

from Problems.Problem import Problem
from Methods.GeneticAlgorithm import GeneticAlgorithm


class FunctionOptimization(Problem):
    def __init__(self, n, limit):
        self.n = n
        self.limit = limit
        self.A = 10

    def func(self, x):
        return -10*self.n - np.sum(np.square(x)-self.A*np.cos(2*np.pi*x))
        # return -np.sum(np.square(np.square(x))-16*np.square(x)+5*x)

    def generate_params(self):
        return np.random.random(size=(self.n, )) * self.limit * 2 - self.limit

    def evaluate(self, param):
        return self.func(param)

    def evaluate_result(self, method):
        if isinstance(method, GeneticAlgorithm):
            best_param = method.param_group[np.argmax(method.param_score)]
            print("Optimized params: ", best_param)
            print("Optimization result: %f" % (self.func(best_param)))
        else:
            print("Optimized params: ", method.params)
            print("Optimization result: %f" % (self.func(method.params)))
