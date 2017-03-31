# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import getopt
import matplotlib.pyplot
import numpy
import sys
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adagrad
from housing_data import ARTData
from housing_data import KingCountyData
from housing_data import NashvilleData
from housing_data import RedfinData

# set RNG with a specific seed
seed = 69
numpy.random.seed(seed)

class MetricTracker(Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.eval_loss = None
        self.eval_acc = None

    def on_epoch_end(self, batch, logs={}):
        self.epoch.append(len(self.epoch))
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

class NeuralNetwork(object):
    def __init__(self, input_shape, num_outputs):
        self.batch_size = 20
        self.adaptive_learning_rate = True
        self.eta_init = 0.1
        self.model = self.create_model(input_shape, num_outputs)
        self.metrics = MetricTracker()
        self.lr_scheduler = LearningRateScheduler(self.update_learning_rate)
        
    def create_model(self, input_shape, num_outputs):
        num_inputs = input_shape[0]
        model = Sequential()
        #model.add(Dense(num_outputs*4, activation='sigmoid', input_shape=input_shape))
        #model.add(Dense(num_outputs*16, activation='sigmoid'))
        model.add(Dense(num_outputs*4, activation='sigmoid', input_shape=input_shape))
        model.add(Dense(num_outputs*2, activation='sigmoid'))
        model.add(Dense(num_outputs, activation='softmax'))
        model.compile(
                optimizer=Adagrad(lr=self.eta_init),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return model
    
    def train(self, data, num_epochs):
        self.model.fit(
                data.X_train, 
                data.y_train,
                nb_epoch=num_epochs,
                batch_size=self.batch_size,
                verbose=1,
                callbacks=[self.metrics, self.lr_scheduler],
                validation_data=(data.X_test, data.y_test))
        
    def test(self, data):
        scores = self.model.evaluate(data.X_test, data.y_test, verbose=1)
        self.metrics.eval_loss = scores[0]
        self.metrics.eval_acc = scores[1]

    def update_learning_rate(self, epoch):
        if epoch > 0 and self.adaptive_learning_rate:
            loss_init = self.metrics.loss[0]
            loss = self.metrics.loss[-1]
            eta = (self.eta_init*(numpy.exp(loss)-1))/(numpy.exp(loss_init)-1)
            print('Epoch {} - Adaptive Learning Rate [eta={:.6f}]'.format(epoch, eta))
        else:
            eta = self.eta_init
        return float(eta)

class Application(object):
    def __init__(self):
        self.num_classes = 5
        self.num_epochs = 50
        
        print('Processing data.')
        self.sources = [NashvilleData('Data/Nashville_geocoded.csv', self.num_classes),
                        KingCountyData('Data/kc_house_data.csv', self.num_classes),
                        RedfinData('Data/redfin.csv', self.num_classes)
                        ARTData('Data/train.csv', self.num_classes)
                       ]
        
    def run(self):
        metrics = []
        for data in self.sources:
            print('{}'.format(data.get_description()))
            
            print('Building neural network.')
            input_shape = data.X_train.shape[1:]
            num_outputs = data.num_classes
            network = NeuralNetwork(input_shape, num_outputs)
            
            print('Training neural network.')
            network.train(data, self.num_epochs)
            
            print('Evaluating neural network.')
            network.test(data)
            print('')
            print('Accuracy: {:.2f}%'.format(network.metrics.eval_acc*100))
            metrics.append(network.metrics)

        self.plot(metrics)
        print('Done.')
        
    def plot(self, metrics):
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.subplot(2, 1, 1)
        matplotlib.pyplot.plot(metrics[0].epoch, metrics[0].acc, 'r', label='Nashville, TN')
        matplotlib.pyplot.plot(metrics[1].epoch, metrics[1].acc, 'g', label='King County, WA')
        matplotlib.pyplot.plot(metrics[2].epoch, metrics[2].acc, 'b', label='Grand Rapids, MI')
        matplotlib.pyplot.plot(metrics[3].epoch, metrics[3].acc, 'c', label='ART data')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Training Accuracy')
        matplotlib.pyplot.legend(loc=4)
        matplotlib.pyplot.subplot(2, 1, 2)
        matplotlib.pyplot.plot(metrics[0].epoch, metrics[0].val_acc, 'r', label='Nashville, TN')
        matplotlib.pyplot.plot(metrics[1].epoch, metrics[1].val_acc, 'g', label='King County, WA')
        matplotlib.pyplot.plot(metrics[2].epoch, metrics[2].val_acc, 'b', label='Grand Rapids, MI')
        matplotlib.pyplot.plot(metrics[3].epoch, metrics[3].val_acc, 'c', label='ART data')
        matplotlib.pyplot.xlabel('Epoch')
        matplotlib.pyplot.ylabel('Testing Accuracy')
        matplotlib.pyplot.legend(loc=4)
        matplotlib.pyplot.savefig('figure_01.jpg')
        matplotlib.pyplot.show()

def main(argv):
    app = Application()
    app.run()

if __name__ == '__main__':
    main(sys.argv[1:])
