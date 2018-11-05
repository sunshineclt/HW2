from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.FunctionOptimization import FunctionOptimization
import numpy as np

fo = FunctionOptimization(n=10)
params = fo.generate_params()


def neighbor_fo(params):
    perturb = np.random.normal(loc=0, scale=1, size=params.shape)
    new_param = params + perturb
    new_param = np.clip(new_param, -5.12, 5.12)
    return new_param


# rhc = RandomHillClimbing(params, neighbor_fo, fo, max_try_per_step=100000)
# rhc.find()
# fo.evaluate_result(rhc)

# sa = SimulatedAnnealing(params, neighbor_fo, fo, max_try_per_step=100000, max_iter=1000, temperature_decay=0.99)
# sa.find()
# fo.evaluate_result(sa)


def crossover(params):
    first, second = params
    ran = np.random.random(size=first.shape)
    offspring1 = np.where(ran < 0.5, first, second)
    offspring2 = np.where(ran > 0.5, first, second)
    return offspring1, offspring2


def mutation(params, p):
    ran = np.random.random(size=params.shape)
    perturb = np.random.normal(loc=0, scale=1, size=params.shape)
    new_params = np.where(ran < p, params + perturb, params)
    new_params = np.clip(new_params, -5.12, 5.12)
    return new_params


population = 100
param_group = []
for i in range(population):
    param_group.append(fo.generate_params())

ga = GeneticAlgorithm(param_group, fo, crossover, mutation, max_iter=10000, mutation_p=0.2)
ga.find()
fo.evaluate_result(ga)
