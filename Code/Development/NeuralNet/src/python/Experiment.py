# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
from HousingData import HousingData
from NeuralNetwork.Activations import Activations
from NeuralNetwork.Dataset import Dataset
from NeuralNetwork.Errors import Errors
from NeuralNetwork.Feedforward import FNN
from NeuralNetwork.Optimizers import SGD
from NeuralNetwork.Optimizers import Adadelta
from NeuralNetwork.Optimizers import Adagrad

class Experiment(object):
    def __init__(self):
        """Creates an application for managing overall execution.
        """
        print('')
        print('Loading data.')
        self.__sources = []
        self.__sources.append(HousingData('../../data/art_processed.csv', name='ART'))
        self.__sources.append(HousingData('../../data/redfin_processed.csv', name='GrandRapids'))
        self.__sources.append(HousingData('../../data/kingcounty_processed.csv', name='KingCounty'))
        self.__sources.append(HousingData('../../data/nashville_processed.csv', name='Nashville'))

        self.__figure = plt.figure(0)
        
    def run(self, num_iters):
        """Executes the application.
        """
        for source in self.__sources:
            self.eval_regress_model(source, [35, 15, 10], Adagrad(learning_rate=0.01), num_iters, label='35_15_10')
            self.eval_regress_model(source, [32, 16, 8], Adagrad(learning_rate=0.01), num_iters, label='35_16_8')
        print('Done.')
    
    def eval_class_model(self, source, hidden_layers, optimizer, num_iters, label='classification'):
        print('')
        print('Data  ' + '-'*60)
        data, targets = source.data
        dataset_train, dataset_test = Dataset(data, targets).split(2.0/3.0)
        self.display_data(source, dataset_train, dataset_test)
         
        print('')
        print('Model ' + '-'*60)
        num_classes = 3
        classes = dataset_train.create_classes(num_classes)
        dataset_train.encode_targets(classes)
        dataset_test.encode_targets(classes)
        layers = [dataset_train.num_features] + hidden_layers + [num_classes]
        network = FNN(layers)
        network.name = label
        network.optimizer = optimizer
        network.activation = Activations.Sigmoid
        network.error = Errors.CategoricalCrossEntropy
        network.match = lambda y, t: np.argmax(y) == np.argmax(t)
        self.display_model(network)

        print('')
        print('Training model.')
        log = network.train(
                        dataset_train,
                        dataset_validate=dataset_test,
                        num_iters=num_iters,
                        batch_size=10,
                        output=self.display_training)
        self.plot(source, network, log)
        self.write_csv(source, network, log)
        
        print('')
        print('Evaluating model.')
        results = network.evaluate(dataset_test)
        self.display_evaluation(results)
    
    def eval_regress_model(self, source, hidden_layers, optimizer, num_iters, label='regression'):
        print('')
        print('Data  ' + '-'*60)
        data, targets = source.data
        dataset_train, dataset_test = Dataset(data, targets).split(2.0/3.0)
        self.display_data(source, dataset_train, dataset_test)
        
        print('')
        print('Model ' + '-'*60)
        layers = [dataset_train.num_features] + hidden_layers + [1]
        network = FNN(layers)
        network.name = label
        network.optimizer = optimizer
        network.activation = Activations.Sigmoid
        network.error = Errors.MeanSquared
        network.match = lambda y, t: np.abs(y - t) <= source.normalize_target(10000)
        self.display_model(network)

        print('')
        print('Training model.')
        log = network.train(
                        dataset_train,
                        dataset_validate=dataset_test,
                        num_iters=num_iters,
                        batch_size=10,
                        output=self.display_training)
        self.plot(source, network, log)
        self.write_csv(source, network, log)
        
        print('')
        print('Evaluating model.')
        results = network.evaluate(dataset_test)
        self.display_evaluation(results)
    
    def display_data(self, source, dataset_train, dataset_test):
        print('Data Source: {}'.format(source.name))
        print('Total features: {}'.format(dataset_train.num_features))
        print('Total entries: {}'.format(dataset_train.num_entries + dataset_test.num_entries))
        print('Training entries: {}'.format(dataset_train.num_entries))
        print('Test entries: {}'.format(dataset_test.num_entries))

    def display_model(self, network):
        print('Type: {}'.format(network.name))
        print('Layers:')
        for layer in network.layers:
            print('\t{}'.format(layer))        
        
    def display_training(self, log):
        iters = log['iters']
        train_losses = log['train_losses']
        train_accs = log['train_accs']
        val_losses = log['val_losses']
        val_accs = log['val_accs']
        
        # Print progress to console.
        print_out = '[{: 4d}] '.format(iters[-1])
        print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(train_losses[-1], train_accs[-1] * 100.0)
        print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(val_losses[-1], val_accs[-1] * 100.0)
        print(print_out)

        # Plot progress.
        if not self.__figure is None:
            self.__figure.clf()
            plt.subplot(2, 1, 1)
            plt.ylabel('Accuracy')
            plt.plot(iters, train_accs, 'r', label='Training Data')
            plt.plot(iters, val_accs, 'g', label='Test Data')
            plt.legend(loc=4)
            plt.subplot(2, 1, 2)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.plot(iters, train_losses, 'r', label='Training Data')
            plt.plot(iters, val_losses, 'g', label='Test Data')
            plt.legend(loc=1)
            plt.pause(0.001)
            plt.draw()

    def display_evaluation(self, results):
        loss = results['loss']
        acc = results['acc']
        print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))

    def write_csv(self, source, network, log):
        filepath = 'results_{}_{}.csv'.format(source.name.lower(), network.name.lower())
        with open(filepath, 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(['iter', 'loss', 'acc', 'val_loss', 'val_acc'])
            iters = log['iters']
            train_losses = log['train_losses']
            train_accs = log['train_accs']
            val_losses = log['val_losses']
            val_accs = log['val_accs']
            rows = [list(r) for r in zip(iters, train_losses, train_accs, val_losses, val_accs)]
            for row in rows:
                writer.writerow(row)

    def plot(self, source, network, log):
        """Plots the given results.
        Arguments
            results - A set of results for the execution of the neural network.
        """
        iters = log['iters']
        train_losses = log['train_losses']
        train_accs = log['train_accs']
        val_losses = log['val_losses']
        val_accs = log['val_accs']
        
        plt.figure(1)
        plt.title(source.name)
        plt.plot(iters, train_accs, 'r', label='Training Data')
        plt.plot(iters, val_accs, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend(loc=4)
        filepath = 'fig_{}_{}_acc.jpg'.format(source.name.lower(), network.name.lower())
        plt.savefig(filepath)
        plt.close()
        
        plt.figure(2)
        plt.title(source.name)
        plt.plot(iters, train_losses, 'r', label='Training Data')
        plt.plot(iters, val_losses, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filepath = 'fig_{}_{}_loss.jpg'.format(source.name.lower(), network.name.lower())
        plt.savefig(filepath)
        plt.close()
        
        plt.figure(3)
        plt.yscale('log')
        plt.title(source.name)
        plt.plot(iters, train_losses, 'r', label='Training Data')
        plt.plot(iters, val_losses, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filepath = 'fig_{}_{}_loss_log.jpg'.format(source.name.lower(), network.name.lower())
        plt.savefig(filepath)
        plt.close()
