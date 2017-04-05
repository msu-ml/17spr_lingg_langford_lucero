# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:10:38 2017

@author: mick
"""

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
        self.sources = [#HousingData('Data/Nashville_processed.csv', name='Nashville, TN'),
                        #HousingData('Data/kingcounty_processed.csv', name='King County, WA'),
                        #HousingData('Data/redfin_processed.csv', name='Grand Rapids, MI'),
                        HousingData('Data/art_processed.csv', name='ART')
                       ]
        
    def run(self):
        """Executes the application.
        """
        num_iters = 1000
        batch_size = 10
        eta = 0.1
        for data in self.sources:
            data_train, data_test = data.split_data(2, 1)
            print('')
            print('Data ' + '-'*60)
            print('Data Source: {}'.format(data.get_name()))
            print('Total features: {}'.format(data.get_num_features()))
            print('Total entries: {}'.format(data.get_num_entries()))
            print('Training entries: {}'.format(len(data_train)))
            print('Test entries: {}'.format(len(data_test)))
            
            """
            n_inputs = data.get_num_features()
            n_hidden = 20
            n_outputs = 10
            classes = data.create_classes(n_outputs)
            data_train = data.classify_targets(data_train, classes)
            data_test = data.classify_targets(data_test, classes)
            network = ClassificationNetwork([n_inputs, n_hidden, n_outputs])
            """
            
            n_inputs = data.get_num_features()
            n_hidden = 10
            n_outputs = 1
            network = RegressionNetwork([n_inputs, n_hidden, n_outputs])
            print('')
            print('Network ' + '-'*60)
            print('Type: Feedforward')
            print('Target Type: Regression')
            print('Layers:')
            print('\t1: {} units'.format(n_inputs))
            print('\t2: {} units'.format(n_hidden))
            print('\t3: {} units'.format(n_outputs))
            
            # Set allowable error for measuring accuracy.
            error = 10000
            y_min = data.unnormalize_target(0.0)
            y_max = data.unnormalize_target(1.0)
            epsilon = error / (y_max - y_min)
            network.set_epsilon(epsilon)
            
            print('')
            print('Training neural network.')
            results = network.train(data_train, data_test, num_iters, batch_size, eta)
            self.plot(data, results)
            
            print('')
            print('Evaluating neural network.')
            loss, acc = network.evaluate(data_test)
            print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))
            
        print('Done.')

    def cross_validate(self):
        candidates = [
                [80],
                [75],
                [70],
                [65],
                [60],
                [75, 50],
                [75, 45],
                [75, 40],
                [75, 35],
                [75, 30],
                [70, 45],
                [70, 40],
                [70, 35],
                [70, 30],
                [70, 25],
                [65, 40],
                [65, 35],
                [65, 30],
                [65, 25],
                [65, 20],
                [70, 45, 25],
                [70, 40, 20],
                [70, 35, 15],
                [70, 30, 10],
                [70, 25, 5]
                ]
        
        num_folds = 5
        num_iters = 10
        batch_size = 10
        eta = 0.1
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
                print('')
                print('Network ' + '-'*60)
                print('Type: Feedforward')
                print('Target Type: Regression')
                print('Layers:')
                for j in xrange(len(layers)):
                    print('\t{}: {} units'.format(j, layers[j]))

                print('')
                print('Cross-validating ({} folds)'.format(num_folds))
                
                best_candidate = None
                best_loss = None
                avg_loss = 0.0
                for i in xrange(num_folds):
                    p = i * fold_size
                    fold_test = data_train[p:p+fold_size]
                    fold_train = data_train[0:p] + data_train[p+fold_size:]
                    
                    results = network.train(fold_train, fold_test, num_iters, batch_size, eta, verbose=False)
                    _, train_loss, _, test_loss, _ = results[-1]
                    print('[{:3d}] Training loss: {:09.6f} Test loss: {:09.6f}'.format(i, train_loss, test_loss))
                    avg_loss += test_loss
                avg_loss /= num_folds
                print('Average loss: {:09.6f}'.format(avg_loss))
                if best_loss == None or best_loss > avg_loss:
                    best_candidate = candidate
                    best_loss = avg_loss
            print('Best candidate: {}'.format(best_candidate))
        
            
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
    experiment = Experiment()
    #experiment.run()
    experiment.cross_validate()

if __name__ == '__main__':
    main(sys.argv[1:])
