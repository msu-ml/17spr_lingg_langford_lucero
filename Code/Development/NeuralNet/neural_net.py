# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy
import sys

class NeuralNetwork(object):
    def __init__(self, layers):
        """Initializes a new neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [numpy.random.randn(b, 1) for b in layers[1:]]
        self.weights = [numpy.random.randn(b, a)
                        for a, b in zip(layers[:-1], layers[1:])]

    def train(self, data_train, num_iters, batch_size, eta, data_validate=None, verbose=True):
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
        # Stochastic Gradient Descent
        results = []
        for i in xrange(num_iters):
            # Randomly shuff the training data.
            numpy.random.shuffle(data_train)
            
            # Divide the training data into batches.
            batches = self.make_batches(data_train, batch_size)
            for batch in batches:
                # Initialize Gradients
                grad_w = [numpy.zeros(w.shape) for w in self.weights]
                grad_b = [numpy.zeros(b.shape) for b in self.biases]
                for x, y in batch:
                    # Compute the gradient for fitting each data point.
                    delta_grad_w, delta_grad_b = self.fit(x, y)
                    grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]
                    grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]

                # Adjust weights.
                self.weights = [w - (eta / len(batch)) * nw
                                for w, nw in zip(self.weights, grad_w)]
                self.biases = [b - (eta / len(batch)) * nb
                               for b, nb in zip(self.biases, grad_b)]
            
            # Evaluate performance on training and test data.
            print_out = '[{:3d}] '.format(i)
            if data_validate == None:
                train_loss, train_acc = self.evaluate(data_train)
                results.append((i, train_loss, train_acc))
                print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(
                                train_loss,
                                train_acc * 100.0)
            else:
                train_loss, train_acc = self.evaluate(data_train)
                test_loss, test_acc = self.evaluate(data_validate)
                results.append((i, train_loss, train_acc, test_loss, test_acc))
                print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(
                                train_loss,
                                train_acc * 100.0)
                print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(
                                test_loss,
                                test_acc * 100.0)
            
            if verbose:
                print(print_out)

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

    def fit(self, x, y):
        """
        Performs a forward pass to compute a prediction and loss value for the
        given data point. Then performs a backward pass to compute the gradient
        for optimizing weights.
        Arguments
            x - A set of data features
            y - An associated target value
        Returns A weight and bias gradient
        """
        return None, None

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        return None

    def evaluate(self, data):
        """Evaluates the loss and accuracy for the model in its current state
        on the given data set.
        Arguments
            data - A set of data to evaluate
        Returns the calculated loss and accuracy.
        """
        return None, None

class ClassificationNetwork(NeuralNetwork):
    def __init__(self, layers):
        """Initializes a new classification neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(ClassificationNetwork, self).__init__(layers)

    def fit(self, x, y):
        """
        Performs a forward pass to compute a prediction and loss value for the
        given data point. Then performs a backward pass to compute the gradient
        for optimizing weights.
        Arguments
            x - A set of data features
            y - An associated target value
        Returns A weight and bias gradient
        """
        grad_w = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]

        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        d_sigmoid = lambda z: sigmoid(z) * (1 - sigmoid(z))
        
        # forward pass
        a = x
        outputs = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = numpy.dot(w, a) + b
            a = sigmoid(z)
            outputs.append(z)
            activations.append(a)
            
        # backward pass
        delta = (activations[-1] - y) * d_sigmoid(outputs[-1])
        grad_w[-1] = numpy.dot(delta, activations[-2].T)
        grad_b[-1] = delta
        for i in xrange(2, self.num_layers):
            delta = numpy.dot(self.weights[-i+1].T, delta) * d_sigmoid(outputs[-i])
            grad_w[-i] = numpy.dot(delta, activations[-i-1].T)
            grad_b[-i] = delta

        return grad_w, grad_b
    
    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        a = x
        for w, b in zip(self.weights, self.biases):
            z = numpy.dot(w, a) + b
            a = sigmoid(z)
        y = a
        return y
    
    def evaluate(self, data):
        """Evaluates the loss and accuracy for the model in its current state
        on the given data set. Uses cross-entropy to measure loss.
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
            
            # compute cross-entropy
            loss += numpy.sum(-numpy.log(py[y==1])) + numpy.sum(-numpy.log(1.0 - py[y==0]))
            
            # check if prediction matches truth
            if numpy.argmax(py) == numpy.argmax(y):
                correct += 1.0
        
        # average metrics across all data points        
        loss = loss / total
        acc = correct / total
        
        return loss, acc

class RegressionNetwork(NeuralNetwork):
    def __init__(self, layers):
        """Initializes a new regression neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(RegressionNetwork, self).__init__(layers)
        self.epsilon = 1e-5

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def fit(self, x, y):
        """
        Performs a forward pass to compute a prediction and loss value for the
        given data point. Then performs a backward pass to compute the gradient
        for optimizing weights.
        Arguments
            x - A set of data features
            y - An associated target value
        Returns A weight and bias gradient
        """
        grad_w = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]

        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        d_sigmoid = lambda z: sigmoid(z) * (1 - sigmoid(z))
        
        # forward pass
        a = x
        outputs = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = numpy.dot(w, a) + b
            a = sigmoid(z)
            outputs.append(z)
            activations.append(a)
            
        # backward pass
        delta = (activations[-1] - y) * d_sigmoid(outputs[-1])
        grad_w[-1] = numpy.dot(delta, activations[-2].T)
        grad_b[-1] = delta
        for i in xrange(2, self.num_layers):
            delta = numpy.dot(self.weights[-i+1].T, delta) * d_sigmoid(outputs[-i])
            grad_w[-i] = numpy.dot(delta, activations[-i-1].T)
            grad_b[-i] = delta

        return grad_w, grad_b

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        a = x
        for w, b in zip(self.weights, self.biases):
            z = numpy.dot(w, a) + b
            a = sigmoid(z)
        y = a[0][0]
        return y

    def evaluate(self, data):
        """Evaluates the loss and accuracy for the model in its current state
        on the given data set. Uses mean-squared-error to measure loss.
        Arguments
            data - A set of data to evaluate
        Returns the calculated loss and accuracy.
        """
        loss = 0.0
        correct = 0.0
        total = len(data)
        for x, y in data:
            y = y[0][0]
            
            # make a prediction for the current data point
            py = self.predict(x)
            
            # compute mean-squared-error
            loss += (py - y)**2
            
            # check if prediction matches truth            
            if abs(py - y) <= self.epsilon:
                correct += 1.0
        
        # average metrics across all data points
        loss = loss / total
        acc = correct / total
        return loss, acc
