from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.FourPeak import FourPeak
import numpy as np

fp = FourPeak(n=50, bonus_threshold=10)
params = fp.generate_params()


def neighbor_fp(params):
    loc = np.random.randint(0, fp.n)
    new_param = params.copy()
    new_param[loc] = 1 - new_param[loc]
    return new_param


# rhc = RandomHillClimbing(params, neighbor_fp, fp, max_try_per_step=100000)
# rhc.find()
# fp.evaluate_result(rhc)

# sa = SimulatedAnnealing(params, neighbor_fp, fp, max_try_per_step=100000, max_iter=1000, initial_temperature=1000, temperature_decay=0.95)
# sa.find()
# fp.evaluate_result(sa)


def crossover(params):
    first, second = params
    cross_point = np.random.randint(1, fp.n)
    offspring1 = np.hstack((first[:cross_point], second[cross_point:]))
    offspring2 = np.hstack((second[:cross_point], first[cross_point:]))
    return offspring1, offspring2


def mutation(params, p):
    ran = np.random.random(size=params.shape)
    update = 1 - params
    new_params = np.where(ran < p, update, params)
    return new_params


population = 100
param_group = []
for i in range(population):
    param_group.append(fp.generate_params())

ga = GeneticAlgorithm(param_group, fp, crossover, mutation, max_iter=-1, mutation_p=0.5)
ga.find(stop_fun=lambda fitness: fitness == fp.max_possible_fit)
fp.evaluate_result(ga)
