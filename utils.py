import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1)[:, np.newaxis]


def acceptance_rate(x, t):
    return np.exp(x / t)


def evaluate_on_all_datasets(problem, method):
    print("Training acc: %f" % (problem.evaluate_on_dataset(method.params, dataset="train") / problem.y_train.shape[0]))
    print("Validation acc: %f" % (problem.evaluate_on_dataset(method.params, dataset="val") / problem.y_val.shape[0]))
    print("Test acc: %f" % (problem.evaluate_on_dataset(method.params, dataset="test") / problem.y_test.shape[0]))
