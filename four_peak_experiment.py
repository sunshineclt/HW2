from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.FourPeak import FourPeak
import numpy as np

fp = FourPeak(n=50, bonus_threshold=20)
params = fp.generate_params()


def neighbor_fp(params):
    loc = np.random.randint(0, fp.n)
    new_param = params
    new_param[loc] = 1 - new_param[loc]
    return new_param


rhc = RandomHillClimbing(params, neighbor_fp, fp, max_try_per_step=100000)
rhc.find()
fp.evaluate_result(rhc)

# sa = SimulatedAnnealing(params, neighbor_fo, fo, max_try_per_step=100000, max_iter=1000, temperature_decay=0.99)
# sa.find()
# fo.evaluate_result(sa)


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
