import numpy as np
import sys, os

sys.path.append(os.pardir)
from activation import *
from layers import *
from loss import *
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, batch_norm=True, weight_decay=0.0):
        self.weight_decay = weight_decay
        self.batch_norm = batch_norm
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        if self.batch_norm:
            self.params['gamma'] = np.random.random(hidden_size)
            self.params['beta'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        if self.batch_norm:
            self.layers['BatchNorm1'] = BatchNormalization(gamma=self.params['gamma'], beta=self.params['beta'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        weight_decay_loss = 0
        for idx in range(1, 3):
            W = self.params['W' + str(idx)]
            weight_decay_loss += 0.5 * self.weight_decay * np.sum(W ** 2)

        y = self.predict(x)
        return self.lastLayer.forward(y, t) + weight_decay_loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout=1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        if self.batch_norm:
            grads['gamma'] = self.layers['BatchNorm1'].dgamma
            grads['beta'] = self.layers['BatchNorm1'].dbeta
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
