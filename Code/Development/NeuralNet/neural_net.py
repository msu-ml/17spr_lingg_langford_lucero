# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import getopt
import matplotlib.pyplot
import numpy
import sys
from housing_data import ARTData
from housing_data import KingCountyData
from housing_data import NashvilleData
from housing_data import RedfinData

# set RNG with a specific seed
seed = 69
numpy.random.seed(seed)

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

    def train(self, data_train, data_test, num_iters, batch_size, eta):
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
        for i in range(num_iters):
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
            train_loss, train_acc = self.evaluate(data_train)
            test_loss, test_acc = self.evaluate(data_test)
            results.append((i, train_loss, train_acc, test_loss, test_acc))
            print('[{:4d}] ' \
                  'training [loss={:.4f} acc={:.2f}] ' \
                  'validating [loss={:.4f} acc={:.2f}]'.format(
                          i,
                          train_loss,
                          train_acc,
                          test_loss,
                          test_acc))
        return results

    def make_batches(self, data, batch_size):
        """Used to create a collection of batches from a set of data.
        Arguments
            data - A data set to divide into batches.
            batch_size - The number of data points in each batch.
        Returns a collection of batches for iteration.
        """
        for i in range(0, len(data), batch_size):
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
        for i in range(2, self.num_layers):
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
        for i in range(2, self.num_layers):
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

class Application(object):
    def __init__(self):
        """Creates an application for managing overall execution.
        """
        print('Processing data.')
        self.sources = [#NashvilleData('Data/Nashville_geocoded.csv'),
                        #KingCountyData('Data/kc_house_data.csv'),
                        #RedfinData('Data/redfin.csv'),
                        ARTData('Data/train.csv')
                       ]
        
    def run(self):
        """Executes the application.
        """
        num_iters = 1000
        batch_size = 10
        eta = 0.1
        for data in self.sources:
            data_train, data_test = data.split_data(2, 1)
            print('Data Source: {}'.format(data.get_name()))
            print('Total features: {}'.format(data.get_num_features()))
            print('Total entries: {}'.format(data.get_num_entries()))
            print('Training entries: {}'.format(len(data_train)))
            print('Test entries: {}'.format(len(data_test)))
            
            print('Creating neural network.')
            #num_inputs = data.get_num_features()
            #num_hidden = 20
            #num_outputs = 10
            #classes = data.create_classes(num_outputs)
            #data_train = data.classify_targets(data_train, classes)
            #data_test = data.classify_targets(data_test, classes)
            #network = ClassificationNetwork([num_inputs, num_hidden, num_outputs])
            
            num_inputs = data.get_num_features()
            num_hidden = 20
            num_outputs = 1
            network = RegressionNetwork([num_inputs, num_hidden, num_outputs])
            
            # Set allowable error for measuring accuracy.
            error = 5000
            y_min = data.unnormalize_target(0.0)
            y_max = data.unnormalize_target(1.0)
            epsilon = error / (y_max - y_min)
            network.set_epsilon(epsilon)
            
            print('Training neural network.')
            results = network.train(data_train, data_test, num_iters, batch_size, eta)
            self.plot(data, results)
            
            print('Evaluating neural network.')
            loss, acc = network.evaluate(data_test)
            print('Results: [loss={:.4f} acc={:.2f}]'.format(loss, acc))
            
        print('Done.')
        
    def plot(self, data, results):
        """Plots the given results.
        Arguments
            results - A set of results for the execution of the neural network.
        """
        iters, train_losses, train_accs, test_losses, test_accs = zip(*results)
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.title(data.get_name())
        matplotlib.pyplot.plot(iters, train_accs, 'r', label='Training Data')
        matplotlib.pyplot.plot(iters, test_accs, 'g', label='Test Data')
        matplotlib.pyplot.xlabel('Iteration')
        matplotlib.pyplot.ylabel('Accuracy')
        matplotlib.pyplot.legend(loc=4)
        matplotlib.pyplot.savefig('fig_accuracy.jpg')
        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.title(data.get_name())
        matplotlib.pyplot.plot(iters, train_losses, 'r', label='Training Data')
        matplotlib.pyplot.plot(iters, test_losses, 'g', label='Test Data')
        matplotlib.pyplot.xlabel('Iteration')
        matplotlib.pyplot.ylabel('Loss')
        matplotlib.pyplot.legend(loc=1)
        matplotlib.pyplot.savefig('fig_loss.jpg')
        matplotlib.pyplot.show()

def main(argv):
    app = Application()
    app.run()

if __name__ == '__main__':
    main(sys.argv[1:])
