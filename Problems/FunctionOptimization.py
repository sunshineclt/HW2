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
        # return -10*self.n - np.sum(np.square(param)-self.A*np.cos(2*np.pi*param))
        # return -(np.sin(param[0]+param[1]) + (param[0]-param[1])**2 - 1.5*param[0]+2.5*param[1])
        # return -(np.square(np.sin(param[0]**2 - param[1]**2))-0.5)/((1+0.001*(param[0]**2+param[1]**2))**2)
        return -(100*(param[1]-param[0])**2-10*param[0]+20*np.cos(param[0]) +
                 100*(param[2]-param[1])**2-10*param[1]+20*np.cos(param[1]))

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
