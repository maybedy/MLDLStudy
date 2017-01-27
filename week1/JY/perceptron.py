import numpy as np

#######################################
# AND, OR, NAND, XOR using PERCEPTRON #
#######################################


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    x = np.array([x1, x2])
    w1 = np.array([[0.5, 0.5], [-0.5, -0.5]])
    b1 = np.array([-0.2, 0.7])
    w2 = np.array([0.5, 0.5])
    b2 = -0.7

    s = np.dot(x, w1) + b1
    s_activated = step_function(s)
    y = np.sum(w2 * s_activated) + b2
    if y <= 0:
        return 0
    else:
        return 1


def test_perceptrons(x):
    print('#####################')
    print('TESTING PERCEPTRONS')
    print('#####################')
    for data in x:
        print("AND(%d,%d): %d" % (data[0], data[1], AND(data[0], data[1])))

    for data in x:
        print("NAND(%d,%d): %d" % (data[0], data[1], NAND(data[0], data[1])))

    for data in x:
        print("OR(%d,%d): %d" % (data[0], data[1], OR(data[0], data[1])))

    for data in x:
        print("XOR(%d,%d): %d" % (data[0], data[1], XOR(data[0], data[1])))
    print('#####################')

if __name__=='__main__':
    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    test_perceptrons(test_data)


