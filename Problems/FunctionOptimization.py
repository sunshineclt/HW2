import numpy as np

from Problems.Problem import Problem
from Methods.GeneticAlgorithm import GeneticAlgorithm


class FunctionOptimization(Problem):
    def __init__(self, n):
        self.n = n
        self.A = 10

    def func(self, x):
        return -10*self.n - np.sum(np.square(x)-self.A*np.cos(2*np.pi*x))

    def generate_params(self):
        return np.random.random(size=(self.n, )) * 5

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
