from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.FunctionOptimization import FunctionOptimization
import numpy as np
import matplotlib.pyplot as plt
from utils import save_result
from datetime import datetime

limit = 4
fo = FunctionOptimization(n=3, limit=limit)


def neighbor_fo(params):
    perturb = np.random.normal(loc=0, scale=1, size=params.shape)
    new_param = params + perturb
    new_param = np.clip(new_param, -limit, limit)
    return new_param


result_sum = 0
result_record = []
experiment_number = 100
start_time = datetime.now()
for i in range(experiment_number):
    params = fo.generate_params()
    rhc = RandomHillClimbing(params, neighbor_fo, fo, max_try_per_step=1, max_iter=4000, print_freq=100)
    result_record.append(rhc.find())
    result = fo.evaluate_result(rhc)
    result_sum += result
print("takes %d seconds" % (datetime.now() - start_time).seconds)
print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
plt.plot(range(100, 4001, 100), np.mean(result_record, axis=0))
plt.savefig("fig/fo_rhc")
plt.show()
save_result(result_record, "fo_rhc")


# result_sum = 0
# result_record = []
# experiment_number = 100
# start_time = datetime.now()
# for i in range(experiment_number):
#     params = fo.generate_params()
#     sa = SimulatedAnnealing(params, neighbor_fo, fo, max_try_per_step=1, max_iter=4000,
#                             initial_temperature=100, temperature_decay=0.997, print_freq=100)
#     result_record.append(sa.find())
#     result = fo.evaluate_result(sa)
#     result_sum += result
# print("takes %d seconds" % (datetime.now() - start_time).seconds)
# print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
# plt.plot(range(100, 4001, 100), np.mean(result_record, axis=0))
# plt.savefig("fig/fo_sa")
# plt.show()
# save_result(result_record, "fo_sa")


# def crossover_pointwise(params):
#     first, second = params
#     ran = np.random.random(size=first.shape)
#     offspring1 = np.where(ran < 0.5, first, second)
#     offspring2 = np.where(ran > 0.5, first, second)
#     return offspring1, offspring2
#
#
# def crossover_crosspoint(params):
#     first, second = params
#     cross_point = np.random.randint(1, fo.n)
#     offspring1 = np.hstack((first[:cross_point], second[cross_point:]))
#     offspring2 = np.hstack((second[:cross_point], first[cross_point:]))
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
# result_sum = 0
# result_record = []
# experiment_number = 100
# start_time = datetime.now()
# for i in range(experiment_number):
#     population = 100
#     param_group = []
#     for _ in range(population):
#         param_group.append(fo.generate_params())
#     ga = GeneticAlgorithm(param_group, fo, crossover_pointwise, mutation, max_iter=2000, print_freq=50, mutation_p=0.1)
#     result_record.append(ga.find())
#     result = fo.evaluate_result(ga)
#     result_sum += result
# print("takes %d seconds" % (datetime.now() - start_time).seconds)
# print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
# plt.plot(range(100, 4001, 100), np.mean(result_record, axis=0), label="pointwise")
# # plt.savefig("fig/fo_ga_pointwise")
# # plt.show()
# save_result(result_record, "fo_ga_pointwise")
#
#
# result_sum = 0
# result_record = []
# experiment_number = 100
# start_time = datetime.now()
# for i in range(experiment_number):
#     population = 100
#     param_group = []
#     for _ in range(population):
#         param_group.append(fo.generate_params())
#     ga = GeneticAlgorithm(param_group, fo, crossover_crosspoint, mutation, max_iter=2000, print_freq=50, mutation_p=0.1)
#     result_record.append(ga.find())
#     result = fo.evaluate_result(ga)
#     result_sum += result
# print("takes %d seconds" % (datetime.now() - start_time).seconds)
# print("Averaged result: %f" % (result_sum * 1.0 / experiment_number))
# plt.plot(range(100, 4001, 100), np.mean(result_record, axis=0), label="crosspoint")
# plt.legend()
# plt.xlabel("iteration")
# plt.ylabel("fitness")
# plt.savefig("fig/fo_ga_crosspoint")
# plt.show()
# save_result(result_record, "fo_ga_crosspoint")
