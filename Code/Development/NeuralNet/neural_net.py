# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy
import sys

class NeuralNetwork(object):
    def __init__(self, layers, dropout=0.0):
        """Initializes a feedforward neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        self.num_layers = len(layers)
        self.layers = layers
        self.dropout = dropout
        self.reset()
        
    def reset(self):
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
        # Stochastic Gradient Descent
        results = []
        delta_W = [0.0 for w in self.weights]
        delta_b = [0.0 for b in self.biases]
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
                delta_W = [gamma*dw + (eta * nw / len(batch))
                            for dw, nw in zip(delta_W, grad_w)]
                delta_b = [gamma*db + (eta * nb / len(batch))
                            for db, nb in zip(delta_b, grad_b)]
                self.weights = [(w - dw) for w, dw in zip(self.weights, delta_W)]
                self.biases = [(b - db) for b, db in zip(self.biases, delta_b)]

            # Evaluate performance on training and test data.
            print_out = '[{:3d}] '.format(i)
            train_loss, train_acc = self.evaluate(data_train)
            test_loss, test_acc = self.evaluate(data_test)
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
        grad_w = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]

        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        d_sigmoid = lambda z: sigmoid(z) * (1 - sigmoid(z))

        # forward pass
        a = x
        outputs = []
        activations = [a]
        masked_weights = []
        for w, b in zip(self.weights, self.biases):
            mask = numpy.random.binomial(1, 1.0 - self.dropout, size=w.shape)
            masked_w  = w * mask

            #z = numpy.dot(w, a) + b            
            z = numpy.dot(masked_w, a) + b
            a = sigmoid(z)
            outputs.append(z)
            activations.append(a)
            masked_weights.append(masked_w)
            
        # backward pass
        delta = (activations[-1] - y) * d_sigmoid(outputs[-1])
        grad_w[-1] = numpy.dot(delta, activations[-2].T)
        grad_b[-1] = delta
        for i in xrange(2, self.num_layers):
            w = self.weights[-i+1]
            masked_w = masked_weights[-i+1]
            
            #delta = numpy.dot(w.T, delta) * d_sigmoid(outputs[-i])
            delta = numpy.dot(masked_w.T, delta) * d_sigmoid(outputs[-i])
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
            adjusted_w = w * (1 - self.dropout)

            #z = numpy.dot(w, a) + b
            z = numpy.dot(adjusted_w, a) + b
            a = sigmoid(z)
        
        return a

    def evaluate(self, data):
        """Evaluates the loss and accuracy for the model in its current state
        on the given data set.
        Arguments
            data - A set of data to evaluate
        Returns the calculated loss and accuracy.
        """
        return None, None

class ClassificationNetwork(NeuralNetwork):
    def __init__(self, layers, dropout=0.0):
        """Initializes a new classification neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(ClassificationNetwork, self).__init__(layers, dropout)
    
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
    def __init__(self, layers, dropout=0.0):
        """Initializes a new regression neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(RegressionNetwork, self).__init__(layers, dropout)
        self.epsilon = 1e-5

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def predict(self, x):
        """Predicts a target value, given a set of data features.
        Arguments
            x - A set of data features
        Returns a target value
        """
        y = super(RegressionNetwork, self).predict(x)
        
        return y[0][0]

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
