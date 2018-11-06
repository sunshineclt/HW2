import numpy as np

from Problems.Problem import Problem
from Methods.GeneticAlgorithm import GeneticAlgorithm


class FunctionOptimization(Problem):
    def __init__(self, n, limit):
        self.n = n
        self.limit = limit
        self.A = 10

    def generate_params(self):
        return np.random.random(size=(self.n, )) * self.limit * 2 - self.limit

    def evaluate(self, param):
        return -10*self.n - np.sum(np.square(param)-self.A*np.cos(2*np.pi*param))

    def evaluate_result(self, method):
        if isinstance(method, GeneticAlgorithm):
            best_param = method.param_group[np.argmax(method.param_score)]
            best_result = self.evaluate(best_param)
            print("Optimized params: ", best_param)
            print("Optimization result: %f" % best_result)
            return best_result
        else:
            print("Optimized params: ", method.params)
            result = self.evaluate(method.params)
            print("Optimization result: %f" % result)
            return result
