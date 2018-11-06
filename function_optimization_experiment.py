from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.FunctionOptimization import FunctionOptimization
import numpy as np
import matplotlib.pyplot as plt
from utils import save_result

limit = 5
fo = FunctionOptimization(n=5, limit=limit)


def neighbor_fo(params):
    perturb = np.random.normal(loc=0, scale=2, size=params.shape)
    new_param = params + perturb
    new_param = np.clip(new_param, -limit, limit)
    return new_param


# result_sum = 0
# result_record = []
# experiment_number = 100
# for i in range(experiment_number):
#     params = fo.generate_params()
#     rhc = RandomHillClimbing(params, neighbor_fo, fo, max_try_per_step=1000, max_iter=100, print_freq=10)
#     result_record.append(rhc.find())
#     result = fo.evaluate_result(rhc)
#     result_sum += result
# print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
# plt.plot(range(10, 101, 10), np.mean(result_record, axis=0))
# plt.show()
# plt.savefig("fig/fo_rhc")
# save_result(result_record, "fo_rhc")


result_sum = 0
result_record = []
experiment_number = 100
for i in range(experiment_number):
    params = fo.generate_params()
    sa = SimulatedAnnealing(params, neighbor_fo, fo, max_try_per_step=1000, max_iter=1000,
                            initial_temperature=100, temperature_decay=0.99, print_freq=100)
    result_record.append(sa.find())
    result = fo.evaluate_result(sa)
    result_sum += result
print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
plt.plot(range(100, 1001, 100), np.mean(result_record, axis=0))
plt.show()
plt.savefig("fig/fo_sa")
save_result(result_record, "fo_sa")


# def crossover(params):
#     first, second = params
#     ran = np.random.random(size=first.shape)
#     offspring1 = np.where(ran < 0.5, first, second)
#     offspring2 = np.where(ran > 0.5, first, second)
#     return offspring1, offspring2
#
#
# def mutation(params, p):
#     ran = np.random.random(size=params.shape)
#     perturb = np.random.normal(loc=0, scale=1, size=params.shape)
#     new_params = np.where(ran < p, params + perturb, params)
#     new_params = np.clip(new_params, -limit, limit)
#     return new_params
#
#
# population = 100
# param_group = []
# for i in range(population):
#     param_group.append(fo.generate_params())
#
# ga = GeneticAlgorithm(param_group, fo, crossover, mutation, max_iter=10000, mutation_p=0.2)
# ga.find()
# fo.evaluate_result(ga)
