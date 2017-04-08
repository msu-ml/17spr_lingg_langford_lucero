# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy
import sys

class NeuralNet(object):
    def __init__(self, layers, name=''):
        """Initializes a feedforward neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        self.layers = layers
        self.name = name
        self.reset()

    def get_layers(self):
        """Gets the layer structure of the network.
        """
        return self.__layers
    def set_layers(self, v):
        """Sets the layer structure of the network.
        """
        self.__layers = v
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
        
    def reset(self):
        """Resets the weights of the network.
        """
        self.biases = [numpy.random.randn(m, 1) for m in self.layers[1:]]
        self.weights = [numpy.random.randn(m, n)
                        for n, m in zip(self.layers[:-1], self.layers[1:])]

    def train(self,
              data_train,
              data_test,
              optimizer=None,
              num_iters=1000,
              batch_size=10,
              learning_rate=0.1,
              momentum=0.0,
              regularization=0.0,
              rho=0.9,
              output=None):
        """Trains the neural network, using Adaptive Gradient Descent (Adagrad)
        for optimizing the model's weights.
        Arguments
            data_train - Data that the model will be fitted to.
            data_test - Data that the model will only be evaluated against.
            optimizer - A gradient descent method for optimization.
            num_iters - The number of iterations to train for.
            batch_size - Number of data points to process in each batch.
            learning_rate - The learning rate for the gradient descent optimizer.
            regularization - Weight regularization term.
            momentum - The momentum for the gradient descent optimizer.
            rho - Parameter for adadelta
            output - A function to display the training progress.
        Returns a collection of performance results.
        """
        best_loss = None
        best_W = self.weights
        best_b = self.biases
        
        if optimizer is None:
            optimizer = self.adadelta
        
        # Adaptive Gradient Descent (Adagrad)
        results = []
        for i in xrange(num_iters):
            # Fit the model to the training data.            
            optimizer(data_train,
                      batch_size,
                      learning_rate=learning_rate,
                      momentum=momentum,
                      regularization=regularization,
                      rho=rho)

            # Evaluate performance on training and test data.
            train_loss, train_acc = self.evaluate(data_train)
            test_loss, test_acc = self.evaluate(data_test)
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                best_W = [numpy.copy(w) for w in self.weights]
                best_b = [numpy.copy(b) for b in self.biases]
            results.append((i, train_loss, train_acc, test_loss, test_acc))
            if not output is None:
                output(results)

        self.weights = best_W
        self.biases = best_b

        return results
    
    def adadelta(self, data, batch_size, learning_rate=1.0, momentum=0.0, regularization=0.0, rho=0.9):
        """Adjusts the models weights to fit the given training data using an
        ADADELTA optimization method.
        """
        eps = 1e-8
        
        # Randomly shuffle the training data and split it into batches.
        numpy.random.shuffle(data)
        batches = self.make_batches(data, batch_size)
            
        # Get gradient for each batch and adjust the weights.
        mem_gW = [numpy.zeros(w.shape) for w in self.weights]
        mem_gb = [numpy.zeros(b.shape) for b in self.biases]
        mem_dW = [numpy.zeros(w.shape) for w in self.weights]
        mem_db = [numpy.zeros(b.shape) for b in self.biases]
        for batch in batches:
            grad_W, grad_b = self.get_batch_gradient(batch)
            mem_gW = [((rho * mgw) + ((1 - rho) * gw**2)) for mgw, gw in zip(mem_gW, grad_W)]
            mem_gb = [((rho * mgb) + ((1 - rho) * gb**2)) for mgb, gb in zip(mem_gb, grad_b)]
            delta_W = [-(gw * numpy.sqrt(mdw + eps) / numpy.sqrt(mgw + eps)) for gw, mdw, mgw in zip(grad_W, mem_dW, mem_gW)]
            delta_b = [-(gb * numpy.sqrt(mdb + eps) / numpy.sqrt(mgb + eps)) for gb, mdb, mgb in zip(grad_b, mem_db, mem_gb)]
            mem_dW = [((rho * mdw) + ((1 - rho) * dw**2)) for mdw, dw in zip(mem_dW, delta_W)]
            mem_db = [((rho * mdb) + ((1 - rho) * db**2)) for mdb, db in zip(mem_db, delta_b)]
            self.weights = [(w + dw) for w, dw in zip(self.weights, delta_W)]
            self.biases = [(b + db) for b, db in zip(self.biases,  delta_b)]
    
    def adagrad(self, data, batch_size, learning_rate=1.0, momentum=0.0, regularization=0.0, rho=0.9):
        """Adjusts the models weights to fit the given training data using an
        ADAGRAD optimization method.
        """
        lr = learning_rate
        reg = regularization
        eps = 1e-8
        
        # Term for regularizing the weights.
        reg_decay = (1.0 - (lr * reg / len(data)))
        
        # Randomly shuffle the training data and split it into batches.
        numpy.random.shuffle(data)
        batches = self.make_batches(data, batch_size)
            
        # Get gradient for each batch and adjust the weights.
        mem_gW = [numpy.zeros(w.shape) for w in self.weights]
        mem_gb = [numpy.zeros(b.shape) for b in self.biases]
        for batch in batches:
            grad_W, grad_b = self.get_batch_gradient(batch)
            mem_gW = [(mw + gw**2) for mw, gw in zip(mem_gW, grad_W)]
            mem_gb = [(mb + gb**2) for mb, gb in zip(mem_gb, grad_b)]
            delta_W = [-(gw * lr / numpy.sqrt(mgw + eps)) for gw, mgw in zip(grad_W, mem_gW)]
            delta_b = [-(gb * lr / numpy.sqrt(mgb + eps)) for gb, mgb in zip(grad_b, mem_gb)]
            self.weights = [(reg_decay * w + dw) for w, dw in zip(self.weights, delta_W)]
            self.biases = [(b + db) for b, db in zip(self.biases,  delta_b)]

    def sgd(self, data, batch_size, learning_rate=1.0, momentum=0.0, regularization=0.0, rho=0.9):
        """Adjusts the models weights to fit the given training data using a
        stochastic gradient descent (SGD) optimization method.
        """
        lr = learning_rate
        reg = regularization
        
        # Term for regularizing the weights.
        reg_decay = (1.0 - (lr * reg / len(data)))
        
        # Randomly shuffle the training data and split it into batches.
        numpy.random.shuffle(data)
        batches = self.make_batches(data, batch_size)
        mem_dW = [numpy.zeros(w.shape) for w in self.weights]
        mem_db = [numpy.zeros(b.shape) for b in self.biases]
        for batch in batches:
            grad_W, grad_b = self.get_batch_gradient(batch)
            delta_W = [((rho * mdw) + (lr * gw)) for mdw, gw in zip(mem_dW, grad_W)]
            delta_b = [((rho * mdb) + (lr * gb)) for mdb, gb in zip(mem_db, grad_b)]
            mem_dW = [dw for dw in delta_W]
            mem_db = [db for db in delta_b]
            #self.weights = [(w - dw) for w, dw in zip(self.weights, delta_W)]
            self.weights = [(reg_decay * w - dw) for w, dw in zip(self.weights, delta_W)]
            self.biases = [(b - db) for b, db in zip(self.biases, delta_b)]

    def gd(self, data, batch_size, learning_rate=1.0, momentum=0.0, regularization=0.0, rho=0.9):
        """Adjusts the models weights to fit the given training data using a
        full batch gradient descent optimization method.
        """
        lr = learning_rate
        grad_W, grad_b = self.get_batch_gradient(data)
        delta_W = [-(lr * gw) for gw in grad_W]
        delta_b = [-(lr * gb) for gb in grad_b]
        self.weights = [(w + dw) for w, dw in zip(self.weights, delta_W)]
        self.biases = [(b + db) for b, db in zip(self.biases, delta_b)]

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
        Returns the average gradient for the given batch.
        """
        batch_grad_W = [numpy.zeros(w.shape) for w in self.weights]
        batch_grad_b = [numpy.zeros(b.shape) for b in self.biases]
        for x, t in batch:
            grad_W, grad_b = self.back_propagation(x, t)
            batch_grad_W = [(bgw + gw) for bgw, gw in zip(batch_grad_W, grad_W)]
            batch_grad_b = [(bgb + gb) for bgb, gb in zip(batch_grad_b, grad_b)]
        batch_grad_W = [(bgw / len(batch)) for bgw in batch_grad_W]
        batch_grad_b = [(bgb / len(batch)) for bgb in batch_grad_b]
        
        return batch_grad_W, batch_grad_b

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
        for i in xrange(self.num_layers-1):
            z = numpy.dot(ws[i], hs[-1]) + bs[i]
            h = self.activation(z)
            zs.append(z)
            hs.append(h)
        y = hs[-1]
            
        # backward pass
        grad_W = [numpy.zeros(w.shape) for w in self.weights]
        grad_b = [numpy.zeros(b.shape) for b in self.biases]
        delta_h = self.error_deriv(y, t)
        for i in xrange(1, self.num_layers):
            delta_h = delta_h * self.activation_deriv(zs[-i])
            grad_W[-i] = numpy.dot(delta_h, hs[-i-1].T)
            grad_b[-i] = delta_h
            delta_h = numpy.dot(ws[-i].T, delta_h)

        return grad_W, grad_b
    
    def activation(self, z):
        """Applies a non-linearity function to determine neuron activation.
        """
        return numpy.nan
    
    def activation_deriv(self, z):
        """Computes the derivative of the activation function.
        """
        return numpy.nan
    
    def error(self, y, t):
        """Computes the error of a prediction using an objective function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a error value for the prediction.
        """
        return numpy.nan
    
    def error_deriv(self, y, t):
        """Computes the derivative of the error function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a value for the derivative of the error.
        """
        return numpy.nan
    
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
        a = x
        for w, b in zip(self.weights, self.biases):
            z = numpy.dot(w, a) + b
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
        for x, t in data:
            # make a prediction for the current data point
            y = self.predict(x)
            
            # compute the error of the prediction
            loss += self.error(y, t)
            
            # check if prediction matches truth
            if self.is_match(y, t):
                correct += 1.0
        
        n = len(data)
        loss = loss / n
        acc = correct / len(data)
        
        return loss, acc

class ClassNet(NeuralNet):
    def __init__(self, layers, name='ClassNet'):
        """Initializes a new classification neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(ClassNet, self).__init__(layers)
    
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
    
    def error(self, y, t):
        """Computes the error of a prediction using cross entropy.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a error value for the prediction.
        """
        #return numpy.sum(-numpy.log(y[t==1])) + numpy.sum(-numpy.log(1.0 - y[t==0]))
        return numpy.sum(-t*numpy.log(y) - (1 - t) * numpy.log(1.0 - y))
    
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
        return numpy.argmax(y) == numpy.argmax(t)


class RegressNet(NeuralNet):
    def __init__(self, layers, name='RegressNet'):
        """Initializes a new regression neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        super(RegressNet, self).__init__(layers)
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

    def error(self, y, t):
        """Computes the error of a prediction using cross entropy.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a error value for the prediction.
        """
        return (y - t[0][0])**2
    
    def error_deriv(self, y, t):
        """Computes the derivative of the error function.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns a value for the derivative of the error.
        """
        return y - t[0][0]
    
    def is_match(self, y, t):
        """Determines if a prediction matches the truth.
        Arguments
            y: A prediction from the network.
            t: The true target value.
        Returns true if the prediction matches the Truth
        """
        return abs(y - t[0][0]) <= self.epsilon
