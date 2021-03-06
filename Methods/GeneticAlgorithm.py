import numpy as np

from Methods.Method import Method
from utils import softmax


class GeneticAlgorithm(Method):
    def __init__(self, param_group, problem, crossover_op, mutation_op, mutation_p=0.1, max_iter=1000, print_freq=1000,
                 verbose=False):
        super().__init__()
        self.param_group = param_group
        self.population = len(self.param_group)
        self.hyper_params = {"max_iter": max_iter}
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.mutation_p = mutation_p
        self.problem = problem
        self.param_score = np.zeros(shape=(self.population, ))
        self.print_freq = print_freq
        self.verbose = verbose

    def step(self):
        sort_index = np.argsort(self.param_score)
        choice = np.random.choice(self.population, size=2, p=softmax(self.param_score), replace=False)
        new_1, new_2 = self.crossover_op((self.param_group[choice[-1]], self.param_group[choice[-2]]))
        new_1 = self.mutation_op(new_1, self.mutation_p)
        new_2 = self.mutation_op(new_2, self.mutation_p)
        new_1_score = self.problem.evaluate(new_1)
        new_2_score = self.problem.evaluate(new_2)

        sort_array = [new_1_score, new_2_score, self.param_score[sort_index[0]], self.param_score[sort_index[1]]]
        sort_param = [new_1, new_2, self.param_group[sort_index[0]], self.param_group[sort_index[1]]]
        sort_new = np.argsort(sort_array)
        is_no_updating = ((sort_new[2] == 2) and (sort_new[3] == 3)) or ((sort_new[3] == 3) and (sort_new[2] == 2))
        if self.verbose:
            if is_no_updating:
                print("No param update! ")
            else:
                print("Replace Score (%f, %f) with (%f, %f)" % (self.param_score[sort_index[0]], self.param_score[sort_index[1]],
                                                                sort_array[sort_new[2]], sort_array[sort_new[3]]))
        self.param_score[sort_index[0]] = sort_array[sort_new[2]]
        self.param_score[sort_index[1]] = sort_array[sort_new[3]]
        self.param_group[sort_index[0]] = sort_param[sort_new[2]]
        self.param_group[sort_index[1]] = sort_param[sort_new[3]]
        return ~is_no_updating

    def find(self, stop_fun=None):
        for index, param in enumerate(self.param_group):
            self.param_score[index] = self.problem.evaluate(param)

        iter_time = 0
        max_iter = self.hyper_params["max_iter"]
        result_record = []
        while max_iter == -1 or iter_time < max_iter:
            iter_time += 1
            if self.verbose:
                print("iter: %d" % iter_time, end=" ")
            self.step()
            if iter_time % self.print_freq == 0:
                best_result = self.problem.evaluate(self.param_group[np.argmax(self.param_score)])
                result_record.append(best_result)
                if self.verbose:
                    print("Iter : %d, best result: %f" % (iter_time, best_result))
                if stop_fun is not None:
                    if stop_fun(best_result):
                        if self.verbose:
                            print("Optimal Reached! ")
                        break
        return result_record
