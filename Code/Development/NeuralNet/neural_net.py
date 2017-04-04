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
        self.num_layers = len(layers)
        self.layers = layers
        self.biases = [numpy.random.randn(b, 1) for b in layers[1:]]
        self.weights = [numpy.random.randn(b, a)
                        for a, b in zip(layers[:-1], layers[1:])]

    def SGD(self, data_train, data_test, num_iters, batch_size, eta):
        results = []
        for i in range(num_iters):
            numpy.random.shuffle(data_train)
            batches = self.make_batches(data_train, batch_size)
            for batch in batches:
                grad_w = [numpy.zeros(w.shape) for w in self.weights]
                grad_b = [numpy.zeros(b.shape) for b in self.biases]
                for x, y in batch:
                    delta_grad_w, delta_grad_b = self.backpropagate(x, y)
                    grad_w = [nw + dnw for nw, dnw in zip(grad_w, delta_grad_w)]
                    grad_b = [nb + dnb for nb, dnb in zip(grad_b, delta_grad_b)]
                self.weights = [w - (eta / len(batch)) * nw
                                for w, nw in zip(self.weights, grad_w)]
                self.biases = [b - (eta / len(batch)) * nb
                               for b, nb in zip(self.biases, grad_b)]
            train_loss, train_acc = self.evaluate(data_train)
            test_loss, test_acc = self.evaluate(data_test)
            results.append((i, train_acc, test_acc))
            print('SGD -[{:4d}]- training [loss={:.4f} acc={:.2f}]  testing [loss={:.4f} acc={:.2f}]'.format(i, train_loss, train_acc, test_loss, test_acc))
        return results

    def make_batches(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    def predict(self, a):
        return None

    def backpropagate(self, x, y):
        return None, None

    def evaluate(self, data):
        return None, None

class ClassificationNetwork(NeuralNetwork):
    def __init__(self, layers, classes):
        self.num_classes = layers[-1]
        self.classes = classes
        super(ClassificationNetwork, self).__init__(layers)
        
    def SGD(self, data_train, data_test, num_iters, batch_size, eta):
        temp_train = zip(*data_train)
        X_train = temp_train[0]
        y_train = [self.target_to_class(y) for y in temp_train[1]]
        data_train = [(X_train[i], y_train[i]) for i in range(len(data_train))]

        temp_test = zip(*data_test)
        X_test = temp_test[0]
        y_test = [self.target_to_class(y) for y in temp_test[1]]
        data_test = [(X_test[i], y_test[i]) for i in range(len(data_test))]
        
        super(ClassificationNetwork, self).SGD(data_train, data_test, num_iters, batch_size, eta)

    def predict(self, a):
        softmax = lambda z: numpy.exp(z) / numpy.sum(numpy.exp(z))
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        for w, b in zip(self.weights, self.biases):
            z = numpy.dot(w, a) + b
            a = sigmoid(z)
        return a

    def backpropagate(self, x, y):
        grad_w = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]

        softmax = lambda z: numpy.exp(z) / numpy.sum(numpy.exp(z))
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
    
    def evaluate(self, data, epsilon=1e-5):
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
    
    def target_to_class(self, target):
        target_class = 0
        for j in range(self.num_classes):
            if target > self.classes[j]:
                target_class = j
        
        t = numpy.zeros((self.num_classes, 1))
        t[target_class] = 1.0

        return t

class RegressionNetwork(NeuralNetwork):
    def __init__(self, layers):
        super(RegressionNetwork, self).__init__(layers)

    def predict(self, a):
        softmax = lambda z: numpy.exp(z) / numpy.sum(numpy.exp(z))
        sigmoid = lambda z: 1.0 / (1.0 + numpy.exp(-z))
        for w, b in zip(self.weights, self.biases):
            y = numpy.dot(w, a) + b
            a = sigmoid(y)
        return a

    def backpropagate(self, x, y):
        grad_w = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]

        softmax = lambda z: numpy.exp(z) / numpy.sum(numpy.exp(z))
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

    def evaluate(self, data, epsilon=1e-5):
        loss = 0.0
        correct = 0.0
        total = len(data)
        for x, y in data:
            # make a prediction for the current data point
            py = self.predict(x)
            
            # compute mean-squared-error
            loss += (py - y)**2
            
            # check if prediction matches truth            
            if abs(py - y) <= epsilon:
                correct += 1.0
        
        # average metrics across all data points
        loss = loss / total
        acc = correct / total
        return loss, acc


class Application(object):
    def __init__(self):
        print('Processing data.')
        self.sources = [#NashvilleData('Data/Nashville_geocoded.csv'),
                        #KingCountyData('Data/kc_house_data.csv'),
                        #RedfinData('Data/redfin.csv'),
                        ARTData('Data/train.csv')
                       ]
        
    def run(self):
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
            num_inputs = data.get_num_features()
            num_hidden = 20
            num_outputs = 10
            classes = data.create_classes(num_outputs)
            network = ClassificationNetwork([num_inputs, num_hidden, num_outputs], classes)
            
            print('Training neural network.')
            results = network.SGD(data_train, data_test, num_iters, batch_size, eta)
            self.plot(results)
            
            print('Evaluating neural network.')
            loss, acc = network.evaluate(data_test)
            print('Results: [loss={:.4f} acc={:.2f}]'.format(loss, acc))
            
        print('Done.')
        
    def plot(self, results):
        iters, train_accs, test_accs = zip(*results)
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.plot(iters, train_accs, 'r', label='Training Data')
        matplotlib.pyplot.plot(iters, test_accs, 'g', label='Test Data')
        matplotlib.pyplot.xlabel('Iteration')
        matplotlib.pyplot.ylabel('Training Accuracy')
        matplotlib.pyplot.legend(loc=4)
        matplotlib.pyplot.savefig('figure_01.jpg')
        matplotlib.pyplot.show()

def main(argv):
    app = Application()
    app.run()

if __name__ == '__main__':
    main(sys.argv[1:])
