# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy
import sys

class NeuralNet(object):
    def __init__(self, layers, dropout=0.0, name=''):
        """Initializes a feedforward neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.dropout = dropout
        self.name = name
        self.reset()
        
    def get_name(self):
        """Gets the name of the network
        Returns the network's name.
        """
        return self.name
        
    def reset(self):
        """Resets the weights of the network.
        """
        self.biases = [numpy.random.randn(m, 1) for m in self.layers[1:]]
        self.weights = [numpy.random.randn(m, n)
                        for n, m in zip(self.layers[:-1], self.layers[1:])]
        
    def train(self, data_train, data_test, num_iters, batch_size, gamma, eta, verbose=True):
        """Trains the neural network, using Stochastic Gradient Descent (SGD)
        for optimizing the model's weights.
        Arguments
            data_train - Data that the model will be fitted to.
            data_test - Data that the model will only be evaluated against.
            num_iters - The number of iterations to train for.
            batch_size - Number of data points to process in each batch.
            eta - The learning rate for adjusting weights.
        Returns a collection of performance results.
        """
        best_loss = None
        best_W = self.weights
        best_b = self.biases
        
        # Stochastic Gradient Descent
        results = []
        for i in xrange(num_iters):
            # Randomly shuffle the training data and split it into batches.
            numpy.random.shuffle(data_train)
            batches = self.make_batches(data_train, batch_size)
            
            # Get gradient for each batch and adjust the weights.
            vW = [numpy.zeros(w.shape) for w in self.weights]
            vb = [numpy.zeros(b.shape) for b in self.biases]
            for batch in batches:
                grad_W, grad_b = self.get_batch_gradient(batch)
                vW = [(gamma * v) - (eta * dw) for v, dw in zip(vW, grad_W)]
                vb = [(gamma * v) - (eta * db) for v, db in zip(vb, grad_b)]
                #vW = [-(eta * dw) for v, dw in zip(vW, grad_W)]
                #vb = [-(eta * db) for v, db in zip(vb, grad_b)]
                self.weights = [(w + v) for w, v in zip(self.weights, vW)]
                self.biases = [(b + v) for b, v in zip(self.biases,  vb)]

            # Evaluate performance on training and test data.
            train_loss, train_acc = self.evaluate(data_train)
            test_loss, test_acc = self.evaluate(data_test)
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                best_W = [numpy.copy(w) for w in self.weights]
                best_b = [numpy.copy(b) for b in self.biases]
            results.append((i, train_loss, train_acc, test_loss, test_acc))
            print_out = '[{:3d}] '.format(i)
            print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(
                            train_loss,
                            train_acc * 100.0)
            print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(
                            test_loss,
                            test_acc * 100.0)
            if verbose:
                print(print_out)

        self.weights = best_W
        self.biases = best_b

        return results
    
    def make_batches(self, data, batch_size):
        """Used to create a collection of batches from a set of data.
        Arguments
            data - A data set to divide into batches.
            batch_size - The number of data points in each batch.
        Returns a collection of batches for iteration.
        """
        for i in xrange(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    def get_batch_gradient(self, batch):
        """ Compute the average gradient for the batch using back propagation.
        Arguments:
            batch - A batch of data.
        Returns the gradient for the given batch.
        """
        grad_W = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]
        for x, y in batch:
            delta_grad_W, delta_grad_b = self.backpropagation(x, y)
            grad_W = [dw + ddw for dw, ddw in zip(grad_W, delta_grad_W)]
            grad_b = [db + ddb for db, ddb in zip(grad_b, delta_grad_b)]
        grad_W = [(dw / len(batch)) for dw in grad_W]
        grad_b = [(db / len(batch)) for db in grad_b]
        
        return grad_W, grad_b

    def backpropagation(self, x, y):
        """
        Performs a forward pass to compute a prediction and loss value for the
        given data point. Then performs a backward pass to compute the gradient
        for optimizing weights.
        Arguments
            x - A set of data features
            y - An associated target value
        Returns A weight and bias gradient
        """
        grad_W = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]

        # forward pass
        a = x
        outputs = []
        activations = [a]
        masked_weights = []
        for w, b in zip(self.weights, self.biases):
            mask = numpy.random.binomial(1, 1.0 - self.dropout, size=w.shape)
            masked_w  = w * mask
            z = numpy.dot(masked_w, a) + b
            a = self.activation(z)
            outputs.append(z)
            activations.append(a)
            masked_weights.append(masked_w)
            
        # backward pass
        delta = self.error_deriv(activations[-1], y) * self.activation_deriv(outputs[-1])
        grad_W[-1] = numpy.dot(delta, activations[-2].T)
        grad_b[-1] = delta
        for i in xrange(2, self.num_layers):
            w = self.weights[-i+1]
            masked_w = masked_weights[-i+1]
            delta = numpy.dot(masked_w.T, delta) * self.activation_deriv(outputs[-i])
            grad_W[-i] = numpy.dot(delta, activations[-i-1].T)
            grad_b[-i] = delta

        return grad_W, grad_b
    
    def activation(self, z):
        """Applies a non-linearity function to determine neuron activation.
        """
        return numpy.nan
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function.
        """
        return numpy.nan
    
    def error(self, py, y):
        """Computes the error of a prediction using an objective function.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns a error value for the prediction.
        """
        return numpy.nan
    
    def error_deriv(self, py, y):
        """Computes the derivative of the error function.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns a value for the derivative of the error.
        """
        return numpy.nan
    
    def is_match(self, py, y):
        """Determines if a prediction matches the truth.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns true if the prediction matches the Truth
        """
        return False

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        a = x
        for w, b in zip(self.weights, self.biases):
            adjusted_w = w * (1 - self.dropout)

            #z = numpy.dot(w, a) + b
            z = numpy.dot(adjusted_w, a) + b
            a = self.activation(z)
        
        return a

    def evaluate(self, data):
        """Evaluates the loss and accuracy for the model in its current state
        on the given data set.
        Arguments
            data - A set of data to evaluate
        Returns the calculated loss and accuracy.
        """
        loss = 0.0
        correct = 0.0
        total = len(data)
        for x, y in data:
            # make a prediction for the current data point
            py = self.predict(x)
            
            # compute the error of the prediction
            loss += self.error(py, y)
            
            # check if prediction matches truth
            if self.is_match(py, y):
                correct += 1.0
        
        # average metrics across all data points        
        loss = loss / total
        acc = correct / total
        
        return loss, acc

class ClassNet(NeuralNet):
    def __init__(self, layers, dropout=0.0, name='ClassNet'):
        """Initializes a new classification neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(ClassNet, self).__init__(layers, dropout)
    
    def activation(self, z):
        """Applies a non-linearity function (sigmoid) to determine neuron
        activation.
        """
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        return sigmoid(z)
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function (sigmoid).
        """
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        return sigmoid(z) * (1 - sigmoid(z))
    
    def error(self, py, y):
        """Computes the error of a prediction using cross entropy.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns a error value for the prediction.
        """
        return numpy.sum(-numpy.log(py[y==1])) + numpy.sum(-numpy.log(1.0 - py[y==0]))
    
    def error_deriv(self, py, y):
        """Computes the derivative of the error function.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns a value for the derivative of the error.
        """
        return (py - y) / ((1 - py) * py)

    def is_match(self, py, y):
        """Determines if a prediction matches the truth.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns true if the prediction matches the Truth
        """
        return numpy.argmax(py) == numpy.argmax(y)


class RegressNet(NeuralNet):
    def __init__(self, layers, dropout=0.0, name='RegressNet'):
        """Initializes a new regression neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(RegressNet, self).__init__(layers, dropout)
        self.epsilon = 1e-5

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        y = super(RegressNet, self).predict(x)
        return y[0][0]
    
    def activation(self, z):
        """Applies a non-linearity function (sigmoid) to determine neuron
        activation.
        """
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        return sigmoid(z)
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function (sigmoid).
        """
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        return sigmoid(z) * (1 - sigmoid(z))

    def error(self, py, y):
        """Computes the error of a prediction using cross entropy.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns a error value for the prediction.
        """
        return (py - y[0][0])**2
    
    def error_deriv(self, py, y):
        """Computes the derivative of the error function.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns a value for the derivative of the error.
        """
        return py - y[0][0]
    
    def is_match(self, py, y):
        """Determines if a prediction matches the truth.
        Arguments
            py: A prediction from the network.
            y: The true target value.
        Returns true if the prediction matches the Truth
        """
        return abs(py - y[0][0]) <= self.epsilon
