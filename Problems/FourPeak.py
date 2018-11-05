import numpy as np

from Problems.Problem import Problem
from Methods.GeneticAlgorithm import GeneticAlgorithm


class FourPeak(Problem):
    def __init__(self, n, bonus_threshold):
        self.n = n
        self.bonus_threshold = bonus_threshold

    def generate_params(self):
        return np.random.randint(0, 2, size=(self.n, ))

    def evaluate(self, param):
        i = 0
        while param[i] == 1:
            i += 1
        head = i
        j = self.n - 1
        while param[j] == 0:
            j -= 1
        tail = self.n - j - 1
        bonus = 0
        if (head > self.bonus_threshold) and (tail > self.bonus_threshold):
            bonus = 100
        return max(head, tail) + bonus

    def evaluate_result(self, method):
        if isinstance(method, GeneticAlgorithm):
            best_param = method.param_group[np.argmax(method.param_score)]
            print("Optimized params: ", best_param)
            print("Optimization result: %f" % (self.evaluate(best_param)))
        else:
            print("Optimized params: ", method.params)
            print("Optimization result: %f" % (self.evaluate(method.params)))
