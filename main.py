from Methods.RandomHillClimbing import RandomHillClimbing
from Problems.NN import NN
import numpy as np

nn = NN()
params = nn.init_nn(150)


def neighbor(params):
    new_params = {}
    for weight_name in params:
        ran = np.random.random(size=params[weight_name].shape)
        perturb = np.random.normal(loc=0, scale=0.1, size=params[weight_name].shape)
        new_value = np.where(ran < 0.1, params[weight_name] + perturb, params[weight_name])
        new_params[weight_name] = new_value
    return new_params


rhc = RandomHillClimbing(params, neighbor, nn, max_try_per_step=10000)
rhc.find()

print("Training acc: %f" % nn.evaluate_on_dataset(rhc.params, dataset="train") / nn.y_train.shape[0])
print("Validation acc: %f" % nn.evaluate_on_dataset(rhc.params, dataset="val") / nn.y_val.shape[0])
print("Test acc: %f" % nn.evaluate_on_dataset(rhc.params, dataset="test") / nn.y_test.shape[0])
