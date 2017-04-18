# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

from NeuralNetwork import NeuralNetwork
import numpy as np
import sys

class FFN(NeuralNetwork):
    def __init__(self, layers):
        super(FFN, self).__init__()
        self.__layers = layers
        self.reset()

    def get_layers(self):
        return self.__layers
    layers = property(fget=lambda self: self.get_layers())

    def get_num_layers(self):
        return len(self.layers)
    num_layers = property(fget=lambda self: self.get_num_layers())

    def reset(self):
        self.weights = []
        self.biases = []
        for a, b in zip(self.layers[:-1], self.layers[1:]):
            self.weights.append(np.random.randn(b, a))
            self.biases.append(np.random.randn(b, 1))

    def back_propagation(self, x, t):
        # forward pass
        ws = [w for w in self.weights]
        bs = [b for b in self.biases]
        zs = []
        hs = [x.reshape((x.shape[0],1))]
        for i in xrange(self.num_layers-1):
            z = np.dot(ws[i], hs[-1]) + bs[i]
            h = self.activation.func(z)
            zs.append(z)
            hs.append(h)
        y = hs[-1]
        t = np.reshape(t, y.shape)

        # backward pass
        grad_W = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        delta_h = self.error.deriv(y, t)
        for i in xrange(1, self.num_layers):
            delta_h = delta_h * self.activation.deriv(zs[-i])
            grad_W[-i] = np.dot(delta_h, hs[-i-1].T)
            grad_b[-i] = delta_h
            delta_h = np.dot(ws[-i].T, delta_h)
        return grad_W, grad_b

    def predict(self, x):
        h = x.reshape((x.shape[0],1))
        for i in xrange(self.num_layers-1):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, h) + b
            h = self.activation.func(z)
        return h