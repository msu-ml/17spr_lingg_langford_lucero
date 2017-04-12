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
from Dataset import Dataset
from NeuralNet import ClassNet
from NeuralNet import RegressNet
from GradientDescent import AdaDelta
from GradientDescent import AdaGrad
from GradientDescent import SGD

class Experiment(object):
    def __init__(self):
        """Creates an application for managing overall execution.
        """
        print('')
        print('Loading data.')
        self.__sources = [#HousingData('Data/nashville_processed.csv', name='Nashville'),
                          #HousingData('Data/kingcounty_processed.csv', name='KingCounty'),
                          #HousingData('Data/redfin_processed.csv', name='GrandRapids'),
                          HousingData('Data/art_processed.csv', name='ART')
                         ]        
        self.__ifigure = plt.figure(0)

    def run(self):
        """Executes the application.
        """
        for source in self.__sources:
            print('')
            print('Data  ' + '-'*60)
            train_dataset, test_dataset = Dataset(*source.data).split(2, 1)
            self.display_data(source, train_dataset, test_dataset)
             
            """Classification network
            print('')
            print('Model ' + '-'*60)
            layers = [train_dataset.num_features, 35, 15, 10, 3]
            network = ClassNet(layers)
            classes = train_dataset.create_classes(3)
            train_dataset.encode_targets(classes)
            test_dataset.encode_targets(classes)
            self.display_model(network)
            """
            
            """Regression network
            """
            print('')
            print('Model ' + '-'*60)
            layers = [train_dataset.num_features, 35, 15, 10, 1]
            network = RegressNet(layers)
            y_min = source.unnormalize_target(0.0)
            y_max = source.unnormalize_target(1.0)
            network.epsilon = 10000 / (y_max - y_min)
            self.display_model(network)
            
            print('')
            print('Training model.')
            plt.ion()
            network.dropout = 0.0
            results = network.train(
                            train_dataset,
                            test_dataset,
                            optimizer=SGD(learning_rate=0.1, momentum=0.9, regularization=0.0),
                            #optimizer=AdaGrad(learning_rate=0.001, regularization=0.5),
                            #optimizer=AdaDelta(scale=0.7),
                            num_iters=2000,
                            batch_size=10,
                            output=self.display_training)
            plt.ioff()
            self.plot(source, network, results)
            
            print('')
            print('Evaluating model.')
            results = network.evaluate(test_dataset)
            self.display_evaluation(results)                            

        print('Done.')
    
    def create_candidate(self):
        n = np.int(np.random.rand() * 9 + 1)
        r = np.random.randint(5, 50, size=(n))
        return r.tolist()
    
    def display_data(self, source, train_dataset, test_dataset):
        print('Data Source: {}'.format(source.name))
        print('Total features: {}'.format(train_dataset.num_features))
        print('Total entries: {}'.format(train_dataset.num_entries + test_dataset.num_entries))
        print('Training entries: {}'.format(train_dataset.num_entries))
        print('Test entries: {}'.format(test_dataset.num_entries))

    def display_model(self, network):
        print('Type: Feedforward Neural Network')
        print('Objective: {}'.format(network.name))
        print('Layers:')
        for i in xrange(network.num_layers):
            print('\t{}: {} units'.format(i, network.layers[i]))
    
    def display_training(self, results):
        if not results is None and len(results) > 0:
            iters, train_losses, train_accs, test_losses, test_accs = zip(*results)
            
            # Print progress to console.
            print_out = '[{: 4d}] '.format(iters[-1])
            print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(train_losses[-1], train_accs[-1] * 100.0)
            print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(test_losses[-1], test_accs[-1] * 100.0)
            print(print_out)
            
            # Plot progress.
            if not self.__ifigure is None:
                self.__ifigure.clf()
                plt.subplot(2, 1, 1)
                plt.ylabel('Accuracy')
                plt.plot(iters, train_accs, 'r', label='Training Data')
                plt.plot(iters, test_accs, 'g', label='Test Data')
                plt.legend(loc=4)
                plt.subplot(2, 1, 2)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.plot(iters, train_losses, 'r', label='Training Data')
                plt.plot(iters, test_losses, 'g', label='Test Data')
                plt.legend(loc=1)
                plt.pause(0.001)
                plt.draw()
    
    def display_evaluation(self, results):
        loss, acc = results
        print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))
    
    def plot(self, source, network, results):
        """Plots the given results.
        Arguments
            results - A set of results for the execution of the neural network.
        """
        iters, train_losses, train_accs, test_losses, test_accs = zip(*results)
        plt.figure(1)
        plt.title(source.name)
        plt.plot(iters, train_accs, 'r', label='Training Data')
        plt.plot(iters, test_accs, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend(loc=4)
        filepath = 'fig_{}_{}_acc.jpg'.format(source.name.lower(), network.name.lower())
        plt.savefig(filepath)
        plt.close()
        plt.figure(2)
        plt.title(source.name)
        plt.plot(iters, train_losses, 'r', label='Training Data')
        plt.plot(iters, test_losses, 'g', label='Test Data')
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
        plt.plot(iters, test_losses, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filepath = 'fig_{}_{}_loss_log.jpg'.format(source.name.lower(), network.name.lower())
        plt.savefig(filepath)
        plt.close()