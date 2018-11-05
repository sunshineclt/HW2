from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.NN import NN
import numpy as np

nn = NN()
params = nn.generate_params(150)


def neighbor_nn(params):
    new_params = {}
    for weight_name in params:
        ran = np.random.random(size=params[weight_name].shape)
        perturb = np.random.normal(loc=0, scale=0.1, size=params[weight_name].shape)
        new_value = np.where(ran < 0.1, params[weight_name] + perturb, params[weight_name])
        new_params[weight_name] = new_value
    return new_params


# rhc = RandomHillClimbing(params, neighbor_nn, nn, max_try_per_step=10000)
# rhc.find()
# nn.evaluate_on_all_datasets(rhc)

# sa = SimulatedAnnealing(params, neighbor_nn, nn, max_try_per_step=10000, max_iter=4000, temperature_decay=0.997)
# sa.find()
# nn.evaluate_on_all_datasets(sa)


def crossover(params):
    first, second = params
    offspring1 = {}
    offspring2 = {}
    for weight_name in first:
        ran = np.random.random(size=first[weight_name].shape)
        offspring1[weight_name] = np.where(ran < 0.5, first[weight_name], second[weight_name])
        offspring2[weight_name] = np.where(ran > 0.5, first[weight_name], second[weight_name])
    return offspring1, offspring2


def mutation(params, p):
    new_params = {}
    for weight_name in params:
        ran = np.random.random(size=params[weight_name].shape)
        perturb = np.random.normal(loc=0, scale=0.1, size=params[weight_name].shape)
        new_value = np.where(ran < p, params[weight_name] + perturb, params[weight_name])
        new_params[weight_name] = new_value
    return new_params


population = 200
param_group = []
for i in range(population):
    param_group.append(nn.generate_params(150))

ga = GeneticAlgorithm(param_group, nn, crossover, mutation, max_iter=5000)
ga.find()
nn.evaluate_on_all_datasets(ga)
