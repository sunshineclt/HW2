from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from utils import evaluate_on_all_datasets
from Problems.NN import NN
import numpy as np

nn = NN()
params = nn.init_nn(150)


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
# evaluate_on_all_datasets(nn, rhc)

sa = SimulatedAnnealing(params, neighbor_nn, nn, max_try_per_step=10000, max_iter=4000, temperature_decay=0.998)
sa.find()
evaluate_on_all_datasets(nn, sa)
