# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy as np
import sys

class GradientDescent(object):
    def __init__(self, learning_rate=1.0):
        """Initializes a new gradient descent optimizer.
        Arguments
            learning_rate - Adjust the intensity of gradient descent.
        """
        self.__learning_rate = learning_rate
    
    def optimize(self, network, dataset, batch_size):
        """Adjusts the models weights to fit the given training data using a
        full batch gradient descent optimization method.
        """
        eta = self.__learning_rate
        grad_W, grad_b = self.get_batch_gradient(network, dataset)
        delta_W = [-(eta * gw) for gw in grad_W]
        delta_b = [-(eta * gb) for gb in grad_b]
        network.weights = [(w + dw) for w, dw in zip(network.weights, delta_W)]
        network.biases = [(b + db) for b, db in zip(network.biases, delta_b)]

    def get_batch_gradient(self, network, batch):
        """ Compute the average gradient for the batch using back propagation.
        Arguments:
            batch - A batch of data.
        Returns the gradient for the given batch.
        """
        batch_grad_W = [np.zeros(w.shape) for w in network.weights]
        batch_grad_b = [np.zeros(b.shape) for b in network.biases]
        for i in xrange(batch.num_entries):
            grad_W, grad_b = network.back_propagation(batch.data[i], batch.targets[i])
            batch_grad_W = [(bgw + gw) for bgw, gw in zip(batch_grad_W, grad_W)]
            batch_grad_b = [(bgb + gb) for bgb, gb in zip(batch_grad_b, grad_b)]
        batch_grad_W = [(bgw / batch.num_entries) for bgw in batch_grad_W]
        batch_grad_b = [(bgb / batch.num_entries) for bgb in batch_grad_b]
        
        return batch_grad_W, batch_grad_b

class SGD(GradientDescent):
    def __init__(self, learning_rate=1.0, momentum=0.0, l1_regularization=0.0, l2_regularization=0.0):
        """Initializes a new schoastic gradient descent optimizer.
        Arguments
            learning_rate - Adjust the intensity of gradient descent.
            momentum - Adjusts the momentum of the gradient descent.
            regularization - Adjusts the amount of L2 regularization for weights.
        """
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__l1_regularization = l1_regularization
        self.__l2_regularization = l2_regularization

    def optimize(self, network, dataset, batch_size):
        """Adjusts the models weights to fit the given training data using a
        stochastic gradient descent (SGD) optimization method.
        """
        eta = self.__learning_rate
        rho = self.__momentum
        lambda1 = self.__l1_regularization
        lambda2 = self.__l2_regularization
        
        # Randomly shuffle the data
        dataset.shuffle()
        
        mem_dW = [np.zeros(w.shape) for w in network.weights]
        mem_db = [np.zeros(b.shape) for b in network.biases]
        batches = dataset.make_batches(batch_size)
        for batch in batches:
            grad_W, grad_b = self.get_batch_gradient(network, batch)
            delta_W = [((rho * mdw) + (eta * gw)) for mdw, gw in zip(mem_dW, grad_W)]
            delta_b = [((rho * mdb) + (eta * gb)) for mdb, gb in zip(mem_db, grad_b)]
            mem_dW = [dw for dw in delta_W]
            mem_db = [db for db in delta_b]
            l1_term = eta * lambda1 / batch_size
            l2_term = eta * lambda2 / batch_size
            network.weights = [(((1.0 - l2_term) * w) - (l1_term * np.sign(w))  - dw) for w, dw in zip(network.weights, delta_W)]
            network.biases = [(b - db) for b, db in zip(network.biases, delta_b)]

class AdaGrad(GradientDescent):
    def __init__(self, learning_rate=1.0):
        """Initializes a new AdaGrad gradient descent optimizer.
        Arguments
            learning_rate - Adjust the intensity of gradient descent.
            regularization - Adjusts the amount of L2 regularization for weights.
        """
        self.__learning_rate = learning_rate

    def optimize(self, network, dataset, batch_size):
        """Adjusts the models weights to fit the given training data using an
        AdaGrad optimization method.
        """
        eta = self.__learning_rate
        eps = 1e-8
        
        # Randomly shuffle the data
        dataset.shuffle()
            
        # Get gradient for each batch and adjust the weights.
        mem_gW = [np.zeros(w.shape) for w in network.weights]
        mem_gb = [np.zeros(b.shape) for b in network.biases]
        batches = dataset.make_batches(batch_size)
        for batch in batches:
            grad_W, grad_b = self.get_batch_gradient(network, batch)
            mem_gW = [(mw + gw**2) for mw, gw in zip(mem_gW, grad_W)]
            mem_gb = [(mb + gb**2) for mb, gb in zip(mem_gb, grad_b)]
            delta_W = [-(gw * eta / np.sqrt(mgw + eps)) for gw, mgw in zip(grad_W, mem_gW)]
            delta_b = [-(gb * eta / np.sqrt(mgb + eps)) for gb, mgb in zip(grad_b, mem_gb)]
            network.weights = [(w + dw) for w, dw in zip(network.weights, delta_W)]
            network.biases = [(b + db) for b, db in zip(network.biases,  delta_b)]

class AdaDelta(GradientDescent):
    def __init__(self, scale=0.9):
        """Initializes a new AdaGrad gradient descent optimizer.
        Arguments
            scale - A term to adjust the affect of the AdaDelta algorithm.
        """
        self.__scale = scale

    def optimize(self, network, dataset, batch_size):
        """Adjusts the models weights to fit the given training data using an
        AdaDelta optimization method.
        """
        rho = self.__scale
        eps = 1e-8
        
        # Randomly shuffle the data
        dataset.shuffle()
            
        # Get gradient for each batch and adjust the weights.
        mem_gW = [np.zeros(w.shape) for w in network.weights]
        mem_gb = [np.zeros(b.shape) for b in network.biases]
        mem_dW = [np.zeros(w.shape) for w in network.weights]
        mem_db = [np.zeros(b.shape) for b in network.biases]
        batches = dataset.make_batches(batch_size)
        for batch in batches:
            grad_W, grad_b = self.get_batch_gradient(network, batch)
            mem_gW = [((rho * mgw) + ((1 - rho) * gw**2)) for mgw, gw in zip(mem_gW, grad_W)]
            mem_gb = [((rho * mgb) + ((1 - rho) * gb**2)) for mgb, gb in zip(mem_gb, grad_b)]
            delta_W = [-(gw * np.sqrt(mdw + eps) / np.sqrt(mgw + eps)) for gw, mdw, mgw in zip(grad_W, mem_dW, mem_gW)]
            delta_b = [-(gb * np.sqrt(mdb + eps) / np.sqrt(mgb + eps)) for gb, mdb, mgb in zip(grad_b, mem_db, mem_gb)]
            mem_dW = [((rho * mdw) + ((1 - rho) * dw**2)) for mdw, dw in zip(mem_dW, delta_W)]
            mem_db = [((rho * mdb) + ((1 - rho) * db**2)) for mdb, db in zip(mem_db, delta_b)]
            network.weights = [(w + dw) for w, dw in zip(network.weights, delta_W)]
            network.biases = [(b + db) for b, db in zip(network.biases,  delta_b)]
