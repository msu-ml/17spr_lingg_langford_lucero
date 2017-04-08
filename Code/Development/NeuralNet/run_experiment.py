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
        print('Processing data.')
        self.sources = [#HousingData('Data/Nashville_processed.csv', name='Nashville'),
                        #HousingData('Data/kingcounty_processed.csv', name='KingCounty'),
                        #HousingData('Data/redfin_processed.csv', name='GrandRapids'),
                        HousingData('Data/art_processed.csv', name='ART')
                       ]
        
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
            n_inputs = data.num_features
            n_hidden1 = 35
            n_hidden2 = 25
            n_hidden3 = 15
            n_outputs = 3
            layers = [n_inputs, n_hidden1, n_hidden2, n_hidden3, n_outputs]
            network = ClassNet(layers)
            classes = data.create_classes(n_outputs)
            data_train = data.classify_targets(data_train, classes)
            data_test = data.classify_targets(data_test, classes)
            """
             
            """Regression network
            """
            print('')
            print('Model ' + '-'*60)
            n_inputs = data.num_features
            n_hidden1 = 35
            n_hidden2 = 25
            n_hidden3 = 15
            n_outputs = 1
            layers = [n_inputs, n_hidden1, n_hidden2, n_hidden3, n_outputs]
            layers = [n_inputs, 100, n_outputs]
            network = RegressNet(layers)
            y_min = data.unnormalize_target(0.0)
            y_max = data.unnormalize_target(1.0)
            network.epsilon = 10000 / (y_max - y_min)
            self.display_model(network)
            
            print('')
            print('Training model.')
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
            self.plot(data, results)
            
            print('')
            print('Evaluating model.')
            results = network.evaluate(data_test)
            self.display_evaluation(results)                            
        print('Done.')

    def cross_validate(self):
        candidates = []
        for i in xrange(100):
            n = np.int(np.random.rand() * 9 + 1)
            r = np.random.randint(5, 50, size=(n))
            candidates.append(r.tolist())
             
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
                                num_iters=200,
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
            i, train_loss, train_acc, test_loss, test_acc = results[-1]
            print_out = '[{: 4d}] '.format(i)
            print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(train_loss, train_acc * 100.0)
            print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(test_loss, test_acc * 100.0)
            print(print_out)
    
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
