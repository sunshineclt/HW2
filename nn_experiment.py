from Methods.RandomHillClimbing import RandomHillClimbing
from Methods.SimulatedAnnealing import SimulatedAnnealing
from Methods.GeneticAlgorithm import GeneticAlgorithm
from Problems.NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import save_result
from datetime import datetime

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


# result_train = []
# result_val = []
# result_test = []
# result_record = []
# experiment_number = 5
# start_time = datetime.now()
# for i in range(experiment_number):
#     params = nn.generate_params(150)
#     rhc = RandomHillClimbing(params, neighbor_nn, nn, max_try_per_step=1, max_iter=50000, print_freq=10, verbose=True)
#     result_record.append(rhc.find())
#     result = nn.evaluate_on_all_datasets(rhc)
#     result_train.append(result[0])
#     result_val.append(result[1])
#     result_test.append(result[2])
# print("takes %d seconds" % (datetime.now() - start_time).seconds)
# print("Averaged train result: %f" % np.mean(result_train))
# print("Averaged val result: %f" % np.mean(result_val))
# print("Averaged test result: %f" % np.mean(result_test))
# plt.plot(range(10, 50001, 10), np.mean(result_record, axis=0))
# plt.savefig("fig/nn_rhc")
# plt.show()
# save_result(result_record, "nn_rhc")

# rhc = RandomHillClimbing(params, neighbor_nn, nn, max_try_per_step=10000)
# rhc.find()
# nn.evaluate_on_all_datasets(rhc)


result_train = []
result_val = []
result_test = []
result_record = []
experiment_number = 3
start_time = datetime.now()
for i in range(experiment_number):
    params = nn.generate_params(150)
    sa = SimulatedAnnealing(params, neighbor_nn, nn, max_try_per_step=1, max_iter=50000,
                            temperature_decay=0.9995, print_freq=10, verbose=True)
    # sa.find(stop_fun=lambda fitness: fitness == fp.max_possible_fit)
    result_record.append(sa.find())
    result = nn.evaluate_on_all_datasets(sa)
    result_train.append(result[0])
    result_val.append(result[1])
    result_test.append(result[2])
print("takes %d seconds" % (datetime.now() - start_time).seconds)
print("Averaged train result: %f" % np.mean(result_train))
print("Averaged val result: %f" % np.mean(result_val))
print("Averaged test result: %f" % np.mean(result_test))
plt.plot(range(10, 50001, 10), np.mean(result_record, axis=0))
plt.savefig("fig/nn_sa_9995")
plt.show()
save_result(result_record, "nn_sa_9995")

# sa = SimulatedAnnealing(params, neighbor_nn, nn, max_try_per_step=10000, max_iter=4000, temperature_decay=0.997)
# sa.find()
# nn.evaluate_on_all_datasets(sa)


# def crossover(params):
#     first, second = params
#     offspring1 = {}
#     offspring2 = {}
#     for weight_name in first:
#         ran = np.random.random(size=first[weight_name].shape)
#         offspring1[weight_name] = np.where(ran < 0.5, first[weight_name], second[weight_name])
#         offspring2[weight_name] = np.where(ran > 0.5, first[weight_name], second[weight_name])
#     return offspring1, offspring2
#
#
# def mutation(params, p):
#     new_params = {}
#     for weight_name in params:
#         ran = np.random.random(size=params[weight_name].shape)
#         perturb = np.random.normal(loc=0, scale=0.1, size=params[weight_name].shape)
#         new_value = np.where(ran < p, params[weight_name] + perturb, params[weight_name])
#         new_params[weight_name] = new_value
#     return new_params
#
#
# result_train = []
# result_val = []
# result_test = []
# result_record = []
# experiment_number = 5
# start_time = datetime.now()
# for i in range(experiment_number):
#     population = 300
#     param_group = []
#     for _ in range(population):
#         param_group.append(nn.generate_params(150))
#     ga = GeneticAlgorithm(param_group, nn, crossover, mutation, max_iter=50000, mutation_p=0.1, print_freq=10, verbose=True)
#     result_record.append(ga.find())
#     result = nn.evaluate_on_all_datasets(ga)
#     result_train.append(result[0])
#     result_val.append(result[1])
#     result_test.append(result[2])
# print("takes %d seconds" % (datetime.now() - start_time).seconds)
# print("Averaged train result: %f" % np.mean(result_train))
# print("Averaged val result: %f" % np.mean(result_val))
# print("Averaged test result: %f" % np.mean(result_test))
# plt.plot(range(10, 50001, 10), np.mean(result_record, axis=0))
# plt.savefig("fig/nn_ga")
# plt.show()
# save_result(result_record, "nn_ga")
