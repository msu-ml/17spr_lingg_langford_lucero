# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy as np
import sys

class NeuralNet(object):
    def __init__(self, layers, name=''):
        """Initializes a feedforward neural network.
        Arguments
            layers - A set of sizes for each layer in the network.
        """
        self.layers = layers
        self.name = name
        self.dropout = 0.0
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
        self.biases = [np.random.randn(b, 1) for b in self.layers[1:]]

    def train(self,
              data_train,
              data_test,
              optimizer,
              num_iters=1000,
              batch_size=10,
              early_stopping = True,
              min_iters=10,
              output=None):
        """Trains the neural network, using Adaptive Gradient Descent (Adagrad)
        for optimizing the model's weights.
        Arguments
            data_train - Data that the model will be fitted to.
            data_test - Data that the model will only be evaluated against.
            optimizer - A gradient descent optimizer.
            num_iters - The number of iterations to train for.
            batch_size - Number of data points to process in each batch.
            early_stopping - If set, training will stop when negligible change is
                             observed in the test loss.
            output - A function to display the training progress.
        Returns a collection of performance results.
        """
        best_loss = None
        best_W = self.weights
        best_b = self.biases
                
        results = []
        for i in xrange(num_iters):
            # Fit the model to the training data.
            optimizer.optimize(self, data_train, batch_size)

            # Evaluate performance on training and test data.
            train_loss, train_acc = self.evaluate(data_train)
            test_loss, test_acc = self.evaluate(data_test)
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                best_W = [np.copy(w) for w in self.weights]
                best_b = [np.copy(b) for b in self.biases]
            results.append((i, train_loss, train_acc, test_loss, test_acc))
            if not output is None:
                output(results)
            
            if early_stopping:
                # Look at the amount of change over the past number of iterations.
                # Stop early if the change is negligible.
                change_window = min_iters
                if len(results) > change_window:
                    change = 0.0
                    for j in xrange(change_window):
                        _, loss1, _, _, _ = results[-j]
                        _, loss2, _, _, _ = results[-j-1]
                        change += loss1 - loss2
                    if np.abs(change) < 1e-6:
                        break;

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

class GradientDescent(object):
    def __init__(self, learning_rate=1.0):
        """Initializes a new gradient descent optimizer.
        Arguments
            learning_rate - Adjust the intensity of gradient descent.
        """
        self.learning_rate = learning_rate
    
    def get_learning_rate(self):
        """Gets the learning rate of the gradient descent.
        """
        return self.__learning_rate
    def set_learning_rate(self, v):
        """Sets the learning rate of the gradient descent.
        """
        self.__learning_rate = v
    learning_rate = property(fget=lambda self: self.get_learning_rate(),
                             fset=lambda self, v: self.set_learning_rate(v))
    
    def optimize(self, network, data, batch_size):
        """Adjusts the models weights to fit the given training data using a
        full batch gradient descent optimization method.
        """
        eta = self.learning_rate
        grad_W, grad_b = self.get_batch_gradient(network, data)
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
        for x, t in batch:
            grad_W, grad_b = network.back_propagation(x, t)
            batch_grad_W = [(bgw + gw) for bgw, gw in zip(batch_grad_W, grad_W)]
            batch_grad_b = [(bgb + gb) for bgb, gb in zip(batch_grad_b, grad_b)]
        batch_grad_W = [(bgw / len(batch)) for bgw in batch_grad_W]
        batch_grad_b = [(bgb / len(batch)) for bgb in batch_grad_b]
        
        return batch_grad_W, batch_grad_b

class SGD(GradientDescent):
    def __init__(self, learning_rate=1.0, momentum=0.0, regularization=0.0):
        """Initializes a new schoastic gradient descent optimizer.
        Arguments
            learning_rate - Adjust the intensity of gradient descent.
            momentum - Adjusts the momentum of the gradient descent.
            regularization - Adjusts the amount of L2 regularization for weights.
        """
        super(SGD, self).__init__(learning_rate=learning_rate)
        self.momentum = momentum
        self.regularization = regularization

    def get_momentum(self):
        """Gets the momentum of the gradient descent.
        """
        return self.__momentum
    def set_momentum(self, v):
        """Sets the momentum of the gradient descent.
        """
        self.__momentum = v
    momentum = property(fget=lambda self: self.get_momentum(),
                        fset=lambda self, v: self.set_momentum(v))

    def get_regularization(self):
        """Gets the amount of L2 regularization for the gradient descent.
        """
        return self.__regularization
    def set_regularization(self, v):
        """Sets the amount of L2 regularization for the gradient descent.
        """
        self.__regularization = v
    regularization = property(fget=lambda self: self.get_regularization(),
                              fset=lambda self, v: self.set_regularization(v))

    def optimize(self, network, data, batch_size):
        """Adjusts the models weights to fit the given training data using a
        stochastic gradient descent (SGD) optimization method.
        """
        eta = self.learning_rate
        rho = self.momentum
        lmbda = self.regularization
        
        # Term for regularizing the weights.
        reg_decay = (1.0 - (eta * lmbda / len(data)))
        
        # Randomly shuffle the training data and split it into batches.
        np.random.shuffle(data)
        
        mem_dW = [np.zeros(w.shape) for w in network.weights]
        mem_db = [np.zeros(b.shape) for b in network.biases]
        for i in xrange(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            grad_W, grad_b = self.get_batch_gradient(network, batch)
            delta_W = [((rho * mdw) + (eta * gw)) for mdw, gw in zip(mem_dW, grad_W)]
            delta_b = [((rho * mdb) + (eta * gb)) for mdb, gb in zip(mem_db, grad_b)]
            mem_dW = [dw for dw in delta_W]
            mem_db = [db for db in delta_b]
            network.weights = [(reg_decay * w - dw) for w, dw in zip(network.weights, delta_W)]
            network.biases = [(b - db) for b, db in zip(network.biases, delta_b)]

class AdaGrad(GradientDescent):
    def __init__(self, learning_rate=1.0, regularization=0.0):
        """Initializes a new AdaGrad gradient descent optimizer.
        Arguments
            learning_rate - Adjust the intensity of gradient descent.
            regularization - Adjusts the amount of L2 regularization for weights.
        """
        super(AdaGrad, self).__init__(learning_rate=learning_rate)
        self.regularization = regularization

    def get_regularization(self):
        """Gets the amount of L2 regularization for the gradient descent.
        """
        return self.__regularization
    def set_regularization(self, v):
        """Sets the amount of L2 regularization for the gradient descent.
        """
        self.__regularization = v
    regularization = property(fget=lambda self: self.get_regularization(),
                              fset=lambda self, v: self.set_regularization(v))

    def optimize(self, network, data, batch_size):
        """Adjusts the models weights to fit the given training data using an
        AdaGrad optimization method.
        """
        eta = self.learning_rate
        lmbda = self.regularization
        eps = 1e-8
        
        # Term for regularizing the weights.
        reg_decay = (1.0 - (eta * lmbda / len(data)))
        
        # Randomly shuffle the training data and split it into batches.
        np.random.shuffle(data)
            
        # Get gradient for each batch and adjust the weights.
        mem_gW = [np.zeros(w.shape) for w in network.weights]
        mem_gb = [np.zeros(b.shape) for b in network.biases]
        for i in xrange(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            grad_W, grad_b = self.get_batch_gradient(network, batch)
            mem_gW = [(mw + gw**2) for mw, gw in zip(mem_gW, grad_W)]
            mem_gb = [(mb + gb**2) for mb, gb in zip(mem_gb, grad_b)]
            delta_W = [-(gw * eta / np.sqrt(mgw + eps)) for gw, mgw in zip(grad_W, mem_gW)]
            delta_b = [-(gb * eta / np.sqrt(mgb + eps)) for gb, mgb in zip(grad_b, mem_gb)]
            network.weights = [(reg_decay * w + dw) for w, dw in zip(network.weights, delta_W)]
            network.biases = [(b + db) for b, db in zip(network.biases,  delta_b)]

class AdaDelta(GradientDescent):
    def __init__(self, scale=0.9):
        """Initializes a new AdaGrad gradient descent optimizer.
        Arguments
            scale - A term to adjust the affect of the AdaDelta algorithm.
        """
        super(AdaDelta, self).__init__(learning_rate=0.0)
        self.scale = scale

    def get_scale(self):
        """Gets the amount to scale the affect of AdaDelta.
        """
        return self.__scale
    def set_scale(self, v):
        """Sets the amount to scale the affect of AdaDelta.
        """
        self.__scale = v
    scale = property(fget=lambda self: self.get_scale(),
                     fset=lambda self, v: self.set_scale(v))

    def optimize(self, network, data, batch_size):
        """Adjusts the models weights to fit the given training data using an
        AdaDelta optimization method.
        """
        rho = self.scale
        eps = 1e-8
        
        # Randomly shuffle the training data and split it into batches.
        np.random.shuffle(data)
            
        # Get gradient for each batch and adjust the weights.
        mem_gW = [np.zeros(w.shape) for w in network.weights]
        mem_gb = [np.zeros(b.shape) for b in network.biases]
        mem_dW = [np.zeros(w.shape) for w in network.weights]
        mem_db = [np.zeros(b.shape) for b in network.biases]
        for i in xrange(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            grad_W, grad_b = self.get_batch_gradient(network, batch)
            mem_gW = [((rho * mgw) + ((1 - rho) * gw**2)) for mgw, gw in zip(mem_gW, grad_W)]
            mem_gb = [((rho * mgb) + ((1 - rho) * gb**2)) for mgb, gb in zip(mem_gb, grad_b)]
            delta_W = [-(gw * np.sqrt(mdw + eps) / np.sqrt(mgw + eps)) for gw, mdw, mgw in zip(grad_W, mem_dW, mem_gW)]
            delta_b = [-(gb * np.sqrt(mdb + eps) / np.sqrt(mgb + eps)) for gb, mdb, mgb in zip(grad_b, mem_db, mem_gb)]
            mem_dW = [((rho * mdw) + ((1 - rho) * dw**2)) for mdw, dw in zip(mem_dW, delta_W)]
            mem_db = [((rho * mdb) + ((1 - rho) * db**2)) for mdb, db in zip(mem_db, delta_b)]
            network.weights = [(w + dw) for w, dw in zip(network.weights, delta_W)]
            network.biases = [(b + db) for b, db in zip(network.biases,  delta_b)]

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
        #return np.sum(-np.log(y[t==1])) + np.sum(-np.log(1.0 - y[t==0]))
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
        return y[0][0]
    
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
