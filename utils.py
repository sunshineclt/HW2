import numpy as np


def softmax_2D(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1)[:, np.newaxis]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def acceptance_rate(x, t):
    return np.exp(x / t)
