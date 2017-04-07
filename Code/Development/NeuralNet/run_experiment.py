# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

import csv
import getopt
import matplotlib.pyplot
import numpy
import sys
from housing_data import HousingData
from neural_net import ClassificationNetwork, RegressionNetwork

# set RNG with a specific seed
seed = 69
numpy.random.seed(seed)

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
        num_iters = 200
        batch_size = 10
        gamma = 0.9
        eta = 0.01
        for data in self.sources:
            data_train, data_test = data.split_data(2, 1)
            print('')
            print('Data ' + '-'*60)
            print('Data Source: {}'.format(data.get_name()))
            print('Total features: {}'.format(data.get_num_features()))
            print('Total entries: {}'.format(data.get_num_entries()))
            print('Training entries: {}'.format(len(data_train)))
            print('Test entries: {}'.format(len(data_test)))
            
            """Classification network
            n_inputs = data.get_num_features()
            n_hidden1 = 35
            n_hidden2 = 25
            n_hidden3 = 15
            n_outputs = 4
            layers = [n_inputs, n_hidden1, n_hidden2, n_hidden3, n_outputs]
            network = ClassificationNetwork(layers, dropout=0.1)
            classes = data.create_classes(n_outputs)
            data_train = data.classify_targets(data_train, classes)
            data_test = data.classify_targets(data_test, classes)
            """
            
            """Regression network
            """
            n_inputs = data.get_num_features()
            n_hidden1 = 35
            n_hidden2 = 25
            n_hidden3 = 15
            n_outputs = 1
            layers = [n_inputs, n_hidden1, n_hidden2, n_hidden3, n_outputs]
            network = RegressionNetwork(layers, dropout=0.0)
            y_min = data.unnormalize_target(0.0)
            y_max = data.unnormalize_target(1.0)
            epsilon = 10000 / (y_max - y_min)
            network.set_epsilon(epsilon)
            
            print('')
            print('Network ' + '-'*60)
            print('Type: Feedforward')
            print('Layers:')
            for j in xrange(len(layers)):
                    print('\t{}: {} units'.format(j, layers[j]))
            
            print('')
            print('Training neural network.')
            results = network.train(data_train, data_test, num_iters, batch_size, gamma, eta)
            self.plot(data, results)
            
            print('')
            print('Evaluating neural network.')
            loss, acc = network.evaluate(data_test)
            print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))
            
            
            """ Test for running different regression nets for different classes.
            data_train.sort(key=lambda (X, y): y)
            data_test.sort(key=lambda d: d[1])
            
            num_classes = 3
            class_size = numpy.int(len(data_train) / num_classes)            
            class_bounds = [data_train[i*class_size+class_size][1]
                                for i in xrange(num_classes)]
            j = 0
            data_train_classes = [[] for _ in xrange(num_classes)]
            for i in xrange(len(data_train)):
                (X, y) = data_train[i]
                if y > class_bounds[j] and j < num_classes - 1:
                    j += 1
                data_train_classes[j].append((X, y))

            j = 0
            data_test_classes = [[] for _ in xrange(num_classes)]                    
            for i in xrange(len(data_test)):
                (X, y) = data_test[i]
                if y > class_bounds[j] and j < num_classes - 1:
                    j += 1
                data_test_classes[j].append((X, y))
            
            for i in xrange(num_classes):
                data_train_class = data_train_classes[i]
                data_test_class = data_test_classes[i]
                
                n_inputs = data.get_num_features()
                n_hidden1 = 35
                n_hidden2 = 25
                n_hidden3 = 15
                n_outputs = 1
                layers = [n_inputs, n_hidden1, n_hidden2, n_hidden3, n_outputs]
                network = RegressionNetwork(layers)
                y_min = data.unnormalize_target(0.0)
                y_max = data.unnormalize_target(1.0)
                epsilon = 10000 / (y_max - y_min)
                network.set_epsilon(epsilon)

                n_inputs = data.get_num_features()
                n_hidden1 = 35
                n_hidden2 = 25
                n_hidden3 = 15
                n_outputs = 1
                layers = [n_inputs, n_hidden1, n_hidden2, n_hidden3, n_outputs]
                network = RegressionNetwork(layers)
                y_min = data.unnormalize_target(0.0)
                y_max = data.unnormalize_target(1.0)
                epsilon = 10000 / (y_max - y_min)
                network.set_epsilon(epsilon)
    
                print('')
                print('Network ' + '-'*60)
                print('Type: Feedforward')
                print('Layers:')
                for j in xrange(len(layers)):
                        print('\t{}: {} units'.format(j, layers[j]))
                
                print('')
                print('Training neural network.')
                results = network.train(data_train_class, data_test_class, num_iters, batch_size, gamma, eta)
                self.plot(data, results)
                
                print('')
                print('Evaluating neural network.')
                loss, acc = network.evaluate(data_test_class)
                print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))
            """
        print('Done.')

    def cross_validate(self):
        candidates = [
                [35],
                [35, 25],
                [35, 25, 15],
                [35, 25, 15, 10],
                [35, 25, 15, 10, 5],
                [25, 15, 10, 5],
                [15, 10, 5],
                [10, 5],
                [5],
                ]                
                
        num_folds = 3
        num_iters = 500
        batch_size = 10
        gamma = 0.9
        eta = 0.01
        record = []
        for data in self.sources:
            data_train, data_test = data.split_data(2, 1)
            print('')
            print('Data    ' + '-'*60)
            print('Data Source: {}'.format(data.get_name()))
            print('Total features: {}'.format(data.get_num_features()))
            print('Total entries: {}'.format(data.get_num_entries()))
            print('Training entries: {}'.format(len(data_train)))
            print('Test entries: {}'.format(len(data_test)))
                
            fold_size = numpy.int(len(data_train) / num_folds)
            for candidate in candidates:
                layers = []
                layers.append(data.get_num_features())
                for value in candidate:
                    layers.append(value)
                layers.append(1)
                network = RegressionNetwork(layers)
                
                # Set allowable error for measuring accuracy.
                error = 10000
                y_min = data.unnormalize_target(0.0)
                y_max = data.unnormalize_target(1.0)
                epsilon = error / (y_max - y_min)
                network.set_epsilon(epsilon)
                
                print('')
                print('Network ' + '-'*60)
                print('Type: Feedforward')
                print('Target Type: Regression')
                print('Layers:')
                for j in xrange(len(layers)):
                    print('\t{}: {} units'.format(j, layers[j]))

                print('')
                print('Cross-validating ({} folds)'.format(num_folds))
                
                avg_loss = 0.0
                for i in xrange(num_folds):
                    p = i * fold_size
                    fold_test = data_train[p:p+fold_size]
                    fold_train = data_train[0:p] + data_train[p+fold_size:]

                    network.reset()                    
                    results = network.train(fold_train, fold_test, num_iters, batch_size, gamma, eta, verbose=False)
                    _, train_loss, train_acc, test_loss, test_acc = results[-1]
                    print_out = '[{}] '.format(i)
                    print_out += 'training [loss={:09.6f} acc={:05.2f}] '.format(
                                    train_loss,
                                    train_acc * 100.0)
                    print_out += 'validating [loss={:09.6f} acc={:05.2f}]'.format(
                                    test_loss,
                                    test_acc * 100.0)
                    print(print_out)
                    avg_loss += test_loss
                avg_loss /= num_folds
                record.append((candidate, avg_loss))
                print('Average loss: {:09.6f}'.format(avg_loss))

            record = numpy.asarray(record)
            record = record[record[:,1].argsort()]
            with open('cross_validate_{}.csv'.format(data.get_name()), 'wb') as out_file:
                writer = csv.writer(out_file)
                writer.writerows(record)
            
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
        matplotlib.pyplot.show()
        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.yscale('log')
        matplotlib.pyplot.title(data.get_name())
        matplotlib.pyplot.plot(iters, train_losses, 'r', label='Training Data')
        matplotlib.pyplot.plot(iters, test_losses, 'g', label='Test Data')
        matplotlib.pyplot.xlabel('Iteration')
        matplotlib.pyplot.ylabel('Loss')
        matplotlib.pyplot.legend(loc=1)
        matplotlib.pyplot.savefig('fig_loss.jpg')
        matplotlib.pyplot.show()

def main(argv):
    experiment = Experiment()
    experiment.run()
    #experiment.cross_validate()

if __name__ == '__main__':
    main(sys.argv[1:])
