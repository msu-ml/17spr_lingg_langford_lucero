# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

import csv
import getopt
import matplotlib.pyplot as plt
import numpy as np
import sys
from housing_data import HousingData
from neural_net import ClassNet
from neural_net import RegressNet

# set RNG with a specific seed
seed = 69
np.random.seed(seed)

class Experiment(object):
    def __init__(self):
        """Creates an application for managing overall execution.
        """
        print('')
        print('Loading data.')
        self.sources = [#HousingData('Data/Nashville_processed.csv', name='Nashville'),
                        #HousingData('Data/kingcounty_processed.csv', name='KingCounty'),
                        #HousingData('Data/redfin_processed.csv', name='GrandRapids'),
                        HousingData('Data/art_processed.csv', name='ART')
                       ]
        
        self.ifigure = plt.figure(0)

    def get_ifigure(self):
        """Gets the interactive plot figure.
        """
        return self.__ifigure
    def set_ifigure(self, v):
        """Sets the interactive plot figure.
        """
        self.__ifigure = v
    ifigure = property(fget=lambda self: self.get_ifigure(),
                       fset=lambda self, v: self.set_ifigure(v))

    def iplot_test(self):
        """Gets the interactive plot for test data.
        """
        return self.__iplot_test
    def set_iplot_test(self, v):
        """Sets the interactive plot for test data.
        """
        self.__iplot_test = v
    iplot_test = property(fget=lambda self: self.get_iplot_test(),
                          fset=lambda self, v: self.set_iplot_test(v))

    def run(self):
        """Executes the application.
        """
        for data in self.sources:
            print('')
            print('Data  ' + '-'*60)
            data_train, data_test = data.split_data(2, 1)
            self.display_data(data, data_train, data_test)
            
            """Classification network
            print('')
            print('Model ' + '-'*60)
            layers = [data.num_features, 100, 1]
            network = ClassNet(layers)
            classes = data.create_classes(n_outputs)
            data_train = data.classify_targets(data_train, classes)
            data_test = data.classify_targets(data_test, classes)
            """
             
            """Regression network
            """
            print('')
            print('Model ' + '-'*60)
            layers = [data.num_features, 100, 1]
            network = RegressNet(layers)
            y_min = data.unnormalize_target(0.0)
            y_max = data.unnormalize_target(1.0)
            network.epsilon = 10000 / (y_max - y_min)
            self.display_model(network)
            
            print('')
            print('Training model.')
            plt.ion()
            results = network.train(
                            data_train,
                            data_test,
                            optimizer=network.sgd,
                            num_iters=1000,
                            batch_size=10,
                            learning_rate=0.1,
                            regularization=0.5,
                            rho=0.9,
                            output=self.display_training)
            plt.ioff()
            self.plot(data, results)
            
            print('')
            print('Evaluating model.')
            results = network.evaluate(data_test)
            self.display_evaluation(results)                            
        print('Done.')

    def cross_validate(self):
        candidates = [self.create_candidate() for _ in xrange(100)]
             
        num_folds = 3
        record = []
        for data in self.sources:
            print('')
            print('Data  ' + '-'*60)
            data_train, data_test = data.split_data(2, 1)
            self.display_data(data, data_train, data_test)
                
            fold_size = np.int(len(data_train) / num_folds)
            for candidate in candidates:
                print('')
                print('Model ' + '-'*60)
                layers = [data.num_features] + candidate + [1]
                network = RegressNet(layers)
                y_min = data.unnormalize_target(0.0)
                y_max = data.unnormalize_target(1.0)
                network.epsilon = 10000 / (y_max - y_min)
                self.display_model(network)

                print('')
                print('Cross-validating ({} folds)'.format(num_folds))
                train_loss_avg = 0.0
                test_loss_avg = 0.0
                for i in xrange(num_folds):
                    p = i * fold_size
                    fold_test = data_train[p:p+fold_size]
                    fold_train = data_train[0:p] + data_train[p+fold_size:]

                    network.reset()                    
                    results = network.train(
                                fold_train,
                                fold_test,
                                optimizer=network.sgd,
                                num_iters=10,
                                batch_size=10,
                                learning_rate=0.1,
                                regularization=0.5,
                                rho=0.9)
                    _, train_loss, train_acc, test_loss, test_acc = results[-1]
                    self.display_training([(i, train_loss, train_acc, test_loss, test_acc)])
                    train_loss_avg += train_loss
                    test_loss_avg += test_loss
                train_loss_avg /= num_folds
                test_loss_avg /= num_folds
                record.append((candidate, train_loss_avg, test_loss_avg))
                print('Average: training [loss={:09.6f}] validating [loss={:09.6f}]'.format(train_loss_avg, test_loss_avg))

            record = np.asarray(record)
            record = record[record[:,2].argsort()]
            with open('cross_validate_{}.csv'.format(data.name), 'wb') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(record)
    
    def create_candidate(self):
        n = np.int(np.random.rand() * 9 + 1)
        r = np.random.randint(5, 50, size=(n))
        return r.tolist()
    
    def display_data(self, data, data_train=None, data_test=None):
        print('Data Source: {}'.format(data.name))
        print('Total features: {}'.format(data.num_features))
        print('Total entries: {}'.format(data.num_entries))
        if not data_train is None:
            print('Training entries: {}'.format(len(data_train)))
        if not data_test is None:
            print('Test entries: {}'.format(len(data_test)))

    def display_model(self, network):
        print('Type: Feedforward Neural Network')
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
            if not self.ifigure is None:
                self.ifigure.clf()
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend(loc=4)
                plt.yscale('log')
                plt.plot(iters, train_losses, 'r', label='Training Data')
                plt.plot(iters, test_losses, 'g', label='Test Data')
                plt.pause(0.001)
                plt.draw()
    
    def display_evaluation(self, results):
        loss, acc = results
        print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))
    
    def plot(self, data, results):
        """Plots the given results.
        Arguments
            results - A set of results for the execution of the neural network.
        """
        iters, train_losses, train_accs, test_losses, test_accs = zip(*results)
        plt.figure(1)
        plt.title(data.name)
        plt.plot(iters, train_accs, 'r', label='Training Data')
        plt.plot(iters, test_accs, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend(loc=4)
        plt.savefig('fig_accuracy.jpg')
        plt.show()
        plt.figure(2)
        plt.title(data.name)
        plt.plot(iters, train_losses, 'r', label='Training Data')
        plt.plot(iters, test_losses, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        plt.savefig('fig_loss.jpg')
        plt.show()
        plt.figure(3)
        plt.yscale('log')
        plt.title(data.name)
        plt.plot(iters, train_losses, 'r', label='Training Data')
        plt.plot(iters, test_losses, 'g', label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        plt.savefig('fig_loss_logscale.jpg')
        plt.show()

def main(argv):
    experiment = Experiment()
    experiment.run()
    #experiment.cross_validate()

if __name__ == '__main__':
    main(sys.argv[1:])
