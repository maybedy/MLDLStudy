import numpy as np
import matplotlib.pylab as plt
import os
clear = lambda: os.system('cls' if os.name=='nt' else 'clear')


# Activation
def step_function(x):
    # if x > 0:
    #     return 1
    # else:
    #     return 0
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x-np.max(x))
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x


# Run test
def plotting(x, y):
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def test(func_str):
    x = np.arange(-5, 5, 0.1)
    y = globals()[func_str](x)
    plotting(x, y)


def test_activation():
    function_list = ['sigmoid', 'relu', 'step_function']
    while True:
        func_str = input('function? ' + ', '.join(function_list) + ' : ')
        if func_str in function_list:
            break
        # clear()
    test(func_str)
test_activation()