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
            results = network.train(data_train, num_iters, batch_size, eta, data_validate=data_test)
            self.plot(data, results)
            
            print('')
            print('Evaluating neural network.')
            loss, acc = network.evaluate(data_test)
            print('Results: [loss={:09.6f} acc={:05.2f}]'.format(loss, acc * 100.0))
            
        print('Done.')

    def cross_validate(self):
        best_eta = None
        best_loss = None
        
        etas = [1e1, 1e-1, 1e-2, 1e-3]
        num_folds = len(etas)
        num_iters = 100
        batch_size = 10
        for data in self.sources:
            data_train, data_test = data.split_data(2, 1)
            print('')
            print('Data    ' + '-'*60)
            print('Data Source: {}'.format(data.get_name()))
            print('Total features: {}'.format(data.get_num_features()))
            print('Total entries: {}'.format(data.get_num_entries()))
            print('Training entries: {}'.format(len(data_train)))
            print('Test entries: {}'.format(len(data_test)))
            
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
            print('Cross-validating')
            fold_size = numpy.int(len(data_train) / num_folds)
            folds = network.make_batches(data_train, fold_size)
            i = 0
            for fold in folds:
                if i < len(etas):
                    eta = etas[i]
                    results = network.train(fold, num_iters, batch_size, eta, verbose=False)
                    _, loss, _ = results[-1]
                    print('[{:3d}] eta={:06.3f} loss={:09.6f}'.format(i, eta, loss))
                    if best_loss == None or best_loss > loss:
                        best_eta = eta
                        best_loss = loss
                    i += 1
        print('Best eta={:06.3f}'.format(best_eta))
        return best_eta
        
            
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
