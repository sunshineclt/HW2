from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.FourPeak import FourPeak
import numpy as np
import matplotlib.pyplot as plt
from utils import save_result

fp = FourPeak(n=50, bonus_threshold=10)
params = fp.generate_params()


def neighbor_fp_single(params):
    loc = np.random.randint(0, fp.n)
    new_param = params.copy()
    new_param[loc] = 1 - new_param[loc]
    return new_param


def neighbor_fp_multi(params):
    ran = np.random.random(size=params.shape)
    update = 1 - params
    new_params = np.where(ran < 0.1, update, params)
    return new_params


result_sum = 0
result_record = []
experiment_number = 100
for i in range(experiment_number):
    rhc = RandomHillClimbing(params, neighbor_fp_multi, fp, max_try_per_step=1000, max_iter=100, print_freq=10)
    result_record.append(rhc.find())
    result = fp.evaluate_result(rhc)
    result_sum += result
print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
plt.plot(range(10, 101, 10), np.mean(result_record, axis=0))
plt.show()
plt.savefig("fig/fp_rhc")
save_result(result_record, "fp_rhc")

# sa = SimulatedAnnealing(params, neighbor_fp_multi, fp, max_try_per_step=100000, max_iter=-1, initial_temperature=1000, temperature_decay=0.95)
# sa.find(stop_fun=lambda fitness: fitness == fp.max_possible_fit)
# fp.evaluate_result(sa)


# def crossover_corsspoint(params):
#     first, second = params
#     cross_point = np.random.randint(1, fp.n)
#     offspring1 = np.hstack((first[:cross_point], second[cross_point:]))
#     offspring2 = np.hstack((second[:cross_point], first[cross_point:]))
#     return offspring1, offspring2
#
#
# def crossover_pointwise(params):
#     first, second = params
#     ran = np.random.random(size=first.shape)
#     offspring1 = np.where(ran < 0.5, first, second)
#     offspring2 = np.where(ran > 0.5, first, second)
#     return offspring1, offspring2
#
#
# def mutation(params, p):
#     ran = np.random.random(size=params.shape)
#     update = 1 - params
#     new_params = np.where(ran < p, update, params)
#     return new_params
#
#
# population = 100
# param_group = []
# for i in range(population):
#     param_group.append(fp.generate_params())
#
# ga = GeneticAlgorithm(param_group, fp, crossover_pointwise, mutation, max_iter=-1, mutation_p=0.1)
# ga.find(stop_fun=lambda fitness: fitness == fp.max_possible_fit)
# fp.evaluate_result(ga)
