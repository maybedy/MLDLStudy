import numpy as np


def identity_function(x):
    return x


def leaky_relu(x):
    return np.max(0.1*x, x)

# TODO test
def relu(x):
    return np.max(0, x) + 2



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x -= np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x -= np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))
