# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

import csv
"""
import matplotlib.pyplot as plt
"""
import numpy as np
import sys
from HousingData import HousingData
from NeuralNetwork.Activations import Activations
from NeuralNetwork.Dataset import Dataset
from NeuralNetwork.Errors import Errors
from NeuralNetwork.Feedforward import FFN
from NeuralNetwork.Optimizers import AdaDelta
from NeuralNetwork.Optimizers import AdaGrad

class Experiment(object):
    def __init__(self):
        """Creates an application for managing overall execution.
        """
        print('')
        print('Loading data.')
        self.__sources = []
        self.__sources.append(HousingData('../../data/art_processed.csv', name='ART'))
        #self.__sources.append(HousingData('../../data/redfin_processed.csv', name='GrandRapids'))
        #self.__sources.append(HousingData('../../data/kingcounty_processed.csv', name='KingCounty'))
        #self.__sources.append(HousingData('../../data/nashville_processed.csv', name='Nashville'))

        """        
        self.__ifigure = plt.figure(0)
        #self.__ifigure = None
        """

    def run(self, num_iters):
        """Executes the application.
        """
        for source in self.__sources:
            print('')
            print('Data  ' + '-'*60)
            dataset_train, dataset_test = Dataset(*source.data).split(2.0/3.0)
            self.display_data(source, dataset_train, dataset_test)
             
            """Classification network
            print('')
            print('Model ' + '-'*60)
            classes = dataset_train.create_classes(3)
            dataset_train.encode_targets(classes)
            dataset_test.encode_targets(classes)
            layers = [dataset_train.num_features, 35, 15, 10, 3]
            network = FFN(layers)
            network.name = 'Classification'
            network.optimizer = SGD(learning_rate=0.1, momentum=0.9)
            network.activation = Activations.Sigmoid
            network.error = Errors.CategoricalCrossEntropy
            network.match = lambda y, t: np.argmax(y) == np.argmax(t)
            self.display_model(network)
            """
            
            """Regression network
            """
            print('')
            print('Model ' + '-'*60)
            layers = [dataset_train.num_features, 32, 16, 1]
            network = FFN(layers)
            network.name = 'Regression'
            network.optimizer = AdaGrad(learning_rate=0.1)
            #network.optimizer = AdaDelta()
            network.activation = Activations.Sigmoid
            network.error = Errors.MeanSquared
            network.match = lambda y, t: np.abs(y - t) <= source.normalize_target(10000)
            self.display_model(network)

            print('')
            print('Training model.')
            """
            plt.ion()
            """
            results = network.train(
                            dataset_train,
                            dataset_validate=dataset_test,
                            num_iters=num_iters,
                            batch_size=10,
                            output=self.display_training)
            """
            plt.ioff()
            """
            self.plot(source, network, results)
            self.write_csv(source, network, results)
            
            print('')
            print('Evaluating model.')
            results = network.evaluate(dataset_test)
            self.display_evaluation(results)                            

        print('Done.')
    
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
        
    def display_training(self, results):
        if not results is None and len(results) > 0:
            iters, train_losses, train_accs, test_losses, test_accs = zip(*results)
            
            # Print progress to console.
            print_out = '[{: 4d}] '.format(iters[-1])
            print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(train_losses[-1], train_accs[-1] * 100.0)
            print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(test_losses[-1], test_accs[-1] * 100.0)
            print(print_out)
            """
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
            """
    def display_evaluation(self, results):
        loss, acc = results
        print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))

    def write_csv(self, source, network, results):
        if not results is None and len(results) > 0:
            filepath = 'results_{}_{}.csv'.format(source.name.lower(), network.name.lower())
            with open(filepath, 'wb') as output_file:
                writer = csv.writer(output_file)
                writer.writerow(['iter', 'loss', 'acc', 'val_loss', 'val_acc'])
                for i in xrange(len(results)):
                    writer.writerow(results[i])

    def plot(self, source, network, results):
        """Plots the given results.
        Arguments
            results - A set of results for the execution of the neural network.
        """
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
        """
