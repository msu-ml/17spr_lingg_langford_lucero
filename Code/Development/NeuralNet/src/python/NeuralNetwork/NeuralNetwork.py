# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import abc
import numpy as np
import sys

class NeuralNetwork(object):
    def __init__(self):
        self.name = ''
        self.weights = []
        self.biases = []
        self.optimizer = None
        self.activation = None
        self.error = None
        self.match = None

    def get_name(self):
        return self.__name
    def set_name(self, v):
        self.__name = v
    name = property(fget=lambda self: self.get_name(),
                    fset=lambda self, v: self.set_name(v))

    def get_weights(self):
        return self.__weights
    def set_weights(self, v):
        self.__weights = v
    weights = property(fget=lambda self: self.get_weights(),
                       fset=lambda self, v: self.set_weights(v))

    def get_biases(self):
        return self.__biases
    def set_biases(self, v):
        self.__biases = v
    biases = property(fget=lambda self: self.get_biases(),
                      fset=lambda self, v: self.set_biases(v))

    def get_optimizer(self):
        return self.__optimizer
    def set_optimizer(self, v):
        self.__optimizer = v
    optimizer = property(fget=lambda self: self.get_optimizer(),
                         fset=lambda self, v: self.set_optimizer(v))

    def get_activation(self):
        return self.__activation 
    def set_activation(self, v):
        self.__activation = v
    activation = property(fget=lambda self: self.get_activation(),
                          fset=lambda self, v: self.set_activation(v))

    def get_error(self):
        return self.__error
    def set_error(self, v):
        self.__error = v
    error = property(fget=lambda self: self.get_error(),
                     fset=lambda self, v: self.set_error(v))
    
    def get_match(self):
        return self.__match
    def set_match(self, v):
        self.__match = v
    match = property(fget=lambda self: self.get_match(),
                     fset=lambda self, v: self.set_match(v))

    @abc.abstractmethod
    def reset(self):
        """Not implemented"""

    def train(self,
              dataset_train,
              dataset_validate=None,
              num_iters=1000,
              batch_size=10,
              output=None):
        results = []
        for i in xrange(num_iters):
            # Fit the model to the training data.
            self.optimizer.optimize(self, dataset_train, batch_size)

            # Evaluate performance on training and test data.
            train_loss, train_acc = self.evaluate(dataset_train)
            val_loss, val_acc = (0.0, 0.0)
            if dataset_validate is not None:
                val_loss, val_acc = self.evaluate(dataset_validate)
            results.append((i, train_loss, train_acc, val_loss, val_acc))
            if not output is None:
                output(results)

        return results
    
    @abc.abstractmethod
    def back_propagation(self, x, t):
        """Not implemented"""
    
    @abc.abstractmethod
    def predict(self, x):
        """Not implemented"""

    def evaluate(self, dataset):
        loss = 0.0
        correct = 0.0
        for i in xrange(dataset.num_entries):
            # make a prediction for the current data point
            y = self.predict(dataset.data[i])
            t = dataset.targets[i]
            t = np.reshape(t, y.shape)

            # compute the error of the prediction
            loss += self.error.func(y, t)
            
            # check if prediction matches truth
            if self.match is not None and self.match(y, t):
                correct += 1.0

        loss = loss / dataset.num_entries
        acc = correct / dataset.num_entries
        
        return loss, acc