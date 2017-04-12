# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy as np
import sys

from Dataset import Dataset
from GradientDescent import AdaDelta
from GradientDescent import AdaGrad
from GradientDescent import GradientDescent
from GradientDescent import SGD

class NeuralNet(object):
    def __init__(self, layers, name=''):
        """Initializes a feedforward neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        self.__layers = layers
        self.name = name
        self.dropout = 0.0
        self.reset()

    def get_layers(self):
        """Gets the layer structure of the network.
        """
        return self.__layers
    layers = property(fget=lambda self: self.get_layers(),
                      fset=lambda self, v: self.set_layers(v))

    def get_num_layers(self):
        """Gets the number of layers in the network.
        """
        return len(self.layers)
    num_layers = property(fget=lambda self: self.get_num_layers())

    def get_weights(self):
        """Gets the weights for the network
        """
        return self.__weights
    def set_weights(self, v):
        """Sets the weights for the network
        """
        self.__weights = v
    weights = property(fget=lambda self: self.get_weights(),
                       fset=lambda self, v: self.set_weights(v))

    def get_biases(self):
        """Gets the biases for the network
        """
        return self.__biases
    def set_biases(self, v):
        """Sets the biases for the network
        """
        self.__biases = v
    biases = property(fget=lambda self: self.get_biases(),
                      fset=lambda self, v: self.set_biases(v))

    def get_name(self):
        """Gets the name of the network.
        """
        return self.__name
    def set_name(self, v):
        """Sets the name of the network.
        """
        self.__name = v
    name = property(fget=lambda self: self.get_name(),
                    fset=lambda self, v: self.set_name(v))
    
    def get_dropout(self):
        """Gets the amount of dropout allowed for each layer.
        """
        return self.__dropout
    def set_dropout(self, v):
        """Sets the amount of dropout allowed for each layer.
        """
        self.__dropout = v
    dropout = property(fget=lambda self: self.get_dropout(),
                       fset=lambda self, v: self.set_dropout(v))
        
    def reset(self):
        """Resets the weights of the network.
        """
        self.weights = [np.random.randn(b, a)
                        for a, b in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.randn(b) for b in self.layers[1:]]

    def train(self,
              train_dataset,
              test_dataset,
              optimizer,
              num_iters=1000,
              batch_size=10,
              output=None):
        """Trains the neural network, using Adaptive Gradient Descent (Adagrad)
        for optimizing the model's weights.
        Arguments
            train_dataset - Data that the model will be fitted to.
            test_dataset - Data that the model will only be evaluated against.
            optimizer - A gradient descent optimizer.
            num_iters - The number of iterations to train for.
            batch_size - Number of data points to process in each batch.
            output - A function to display the training progress.
        Returns a collection of performance results.
        """
        best_loss = None
        best_W = self.weights
        best_b = self.biases
                
        results = []
        for i in xrange(num_iters):
            # Fit the model to the training data.
            optimizer.optimize(self, train_dataset, batch_size)

            # Evaluate performance on training and test data.
            train_loss, train_acc = self.evaluate(train_dataset)
            test_loss, test_acc = self.evaluate(test_dataset)
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                best_W = [np.copy(w) for w in self.weights]
                best_b = [np.copy(b) for b in self.biases]
            results.append((i, train_loss, train_acc, test_loss, test_acc))
            if not output is None:
                output(results)

        self.weights = best_W
        self.biases = best_b

        return results

    def back_propagation(self, x, t):
        """
        Performs a forward pass to compute a prediction and loss value for the
        given data point. Then performs a backward pass to compute the change
        in gradient for optimizing weights.
        Arguments
            x - A set of data features
            t - A target value
        Returns a weight and bias gradient for the given data point.
        """
        # forward pass
        ws = [w for w in self.weights]
        bs = [b for b in self.biases]
        zs = []
        hs = [x]
        masks = []
        for i in xrange(self.num_layers-1):
            z = np.dot(ws[i], hs[-1]) + bs[i]
            h = self.activation(z)

            # apply dropout for each layer but the last
            if i < self.num_layers-2:
                mask = np.random.binomial(1.0, 1.0 - self.dropout, size=h.shape)
            else:
                mask = np.ones(h.shape)
            masks.append(mask)
            h *= mask
            
            zs.append(z)
            hs.append(h)
        y = hs[-1]

        # backward pass
        grad_W = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        delta_h = self.error_deriv(y, t)
        for i in xrange(1, self.num_layers):
            delta_h = delta_h * self.activation_deriv(zs[-i])
            delta_h *= masks[-i]
            if delta_h.ndim < 2:
                grad_W[-i] = np.outer(delta_h, hs[-i-1].T)
            else:
                grad_W[-i] = np.dot(delta_h, hs[-i-1].T)
            grad_b[-i] = delta_h
            delta_h = np.dot(ws[-i].T, delta_h)

        return grad_W, grad_b
    
    def activation(self, z):
        """Applies a non-linearity function to determine neuron activation.
        """
        return np.nan
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function.
        """
        return np.nan
    
    def error(self, y, t):
        """Computes the error of a prediction using an objective function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a error value for the prediction.
        """
        return np.nan
    
    def error_deriv(self, y, t):
        """Computes the derivative of the error function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a value for the derivative of the error.
        """
        return np.nan
    
    def is_match(self, y, t):
        """Determines if a prediction matches the truth.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns true if the prediction matches the Truth
        """
        return False

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        h = x
        for i in xrange(self.num_layers-1):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, h) + b
            h = self.activation(z)
            
            # factor in dropout
            if i < self.num_layers-2:
                h *= (1.0 - self.dropout)

        return h

    def evaluate(self, dataset):
        """Evaluates the loss and accuracy for the model in its current state
        on the given data set.
        Arguments
            data - A set of data to evaluate
        Returns the calculated loss and accuracy.
        """
        loss = 0.0
        correct = 0.0
        for i in xrange(dataset.num_entries):
            # make a prediction for the current data point
            y = self.predict(dataset.data[i])
            
            # compute the error of the prediction
            loss += self.error(y, dataset.targets[i])
            
            # check if prediction matches truth
            if self.is_match(y, dataset.targets[i]):
                correct += 1.0
        
        loss = loss / dataset.num_entries
        acc = correct / dataset.num_entries
        
        return loss, acc

class ClassNet(NeuralNet):
    def __init__(self, layers, name='Classification'):
        """Initializes a new classification neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(ClassNet, self).__init__(layers, name)
    
    def activation(self, z):
        """Applies a non-linearity function (sigmoid) to determine neuron
        activation.
        """
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        return sigmoid(z)
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function (sigmoid).
        """
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        return sigmoid(z) * (1 - sigmoid(z))
    
    def error(self, y, t):
        """Computes the error of a prediction using cross entropy.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a error value for the prediction.
        """
        return np.sum(-t*np.log(y) - (1 - t) * np.log(1.0 - y))
    
    def error_deriv(self, y, t):
        """Computes the derivative of the error function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a value for the derivative of the error.
        """
        return (y - t)
    
    def regularized_error(self, w, y, t):
        return self.error(y, t)

    def is_match(self, y, t):
        """Determines if a prediction matches the truth.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns true if the prediction matches the Truth
        """
        return np.argmax(y) == np.argmax(t)


class RegressNet(NeuralNet):
    def __init__(self, layers, name='Regression'):
        """Initializes a new regression neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(RegressNet, self).__init__(layers, name)
        self.epsilon = 1e-5

    def get_epsilon(self):
        """Gets the allowable error, epsilon, for measuring prediction accuracy.
        """
        return self.__epsilon
    def set_epsilon(self, v):
        """Sets the allowable error, epsilon, for measuring prediction accuracy.
        """
        self.__epsilon = v
    epsilon = property(fget=lambda self: self.get_epsilon(),
                       fset=lambda self, v: self.set_epsilon(v))

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        y = super(RegressNet, self).predict(x)
        return y
    
    def activation(self, z):
        """Applies a non-linearity function (sigmoid) to determine neuron
        activation.
        """
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        return sigmoid(z)
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function (sigmoid).
        """
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
        return sigmoid(z) * (1.0 - sigmoid(z))

    def error(self, y, t):
        """Computes the error of a prediction using cross entropy.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a error value for the prediction.
        """
        return np.sum((y - t)**2)
    
    def error_deriv(self, y, t):
        """Computes the derivative of the error function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a value for the derivative of the error.
        """
        return np.sum(y - t)
    
    def is_match(self, y, t):
        """Determines if a prediction matches the truth.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns true if the prediction matches the Truth
        """
        return np.all(abs(y - t) <= self.epsilon)