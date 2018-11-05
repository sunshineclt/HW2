from Methods.Method import Method
from utils import acceptance_rate
import numpy as np


class SimulatedAnnealing(Method):
    def __init__(self, params, neighbor_op, problem, max_try_per_step=100, max_iter=1000, initial_temperature=100, temperature_decay=0.99):
        super().__init__()
        self.params = params
        self.hyper_params = {"max_try_per_step": max_try_per_step,
                             "max_iter": max_iter}
        self.neighbor_op = neighbor_op
        self.problem = problem
        self.previous_score = 0
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.temperature = 0

    def step(self):
        time = 0
        while time < self.hyper_params["max_try_per_step"]:
            neighbor = self.neighbor_op(self.params)
            neighbor_score = self.problem.evaluate(neighbor)
            if (self.previous_score < neighbor_score) or (np.random.random() < acceptance_rate(neighbor_score - self.previous_score, self.temperature)):
                self.params = neighbor
                print("Param Updated! Score %f -> %f" % (self.previous_score, neighbor_score))
                self.previous_score = neighbor_score
                break
            time += 1
        is_no_updating = time == self.hyper_params["max_try_per_step"]
        if is_no_updating:
            print("No param update! ")
        return time != self.hyper_params["max_try_per_step"]

    def find(self):
        iter_time = 0
        is_updating = True
        self.previous_score = self.problem.evaluate(self.params)
        max_iter = self.hyper_params["max_iter"]
        self.temperature = self.initial_temperature
        while is_updating and (max_iter == -1 or iter_time < max_iter):
            iter_time += 1
            print("iter: %d" % iter_time, end=" ")
            is_updating = self.step()
            self.temperature *= self.temperature_decay
