import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1)[:, np.newaxis]


def acceptance_rate(x, t):
    return np.exp(x / t)
