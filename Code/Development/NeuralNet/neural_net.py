# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import getopt
import numpy
import sys
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from housing_data import KingCountyData
from housing_data import NashvilleData
from housing_data import RedfinData

# set RNG with a specific seed
seed = 69
numpy.random.seed(seed)

class LossTracker(Callback):
    def on_train_begin(self, logs={}):
        self.loss_init = -1
        self.loss = -1
    def on_epoch_end(self, batch, logs={}):
        self.loss = logs.get('loss')
        if self.loss_init < 0:
            self.loss_init = self.loss

class NeuralNetwork(object):
    def __init__(self, input_shape, num_outputs):
        self.adaptive_learning_rate = False
        self.eta_init = 0.1
        self.batch_size = 20
        self.model = self.create_model(input_shape, num_outputs)
        self.loss_tracker = LossTracker()
        self.lr_scheduler = LearningRateScheduler(self.update_learning_rate)
        
    def create_model(self, input_shape, num_outputs):
        model = Sequential()
        model.add(Dense(num_outputs*4, activation='sigmoid', input_shape=input_shape))
        model.add(Dense(num_outputs*2, activation='sigmoid'))
        model.add(Dense(num_outputs, activation='softmax'))
        model.compile(
                optimizer=RMSprop(lr=self.eta_init),
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
                callbacks=[self.loss_tracker, self.lr_scheduler],
                validation_data=(data.X_test, data.y_test))
        
    def test(self, data):
        scores = self.model.evaluate(data.X_test, data.y_test, verbose=1)
        return scores[1]

    def update_learning_rate(self, epoch):
        if epoch > 0 and self.adaptive_learning_rate:
            loss_init = self.loss_tracker.loss_init
            loss = self.loss_tracker.loss
            eta = (self.eta_init*(numpy.exp(loss)-1))/(numpy.exp(loss_init)-1)
            print('Epoch {} - Adaptive Learning Rate [eta={:.6f}]'.format(epoch, eta))
        else:
            eta = self.eta_init
        return float(eta)

class Application(object):
    def __init__(self):
        target_dist = [0, 70000, 100000, 125000, 140000, 160000, 180000, 210000, 250000, 325000]
        
        print('Loading data.')
        self.data = NashvilleData('Data/Nashville_housing_data_2013_2016.csv', target_dist)
        #self.data = KingCountyData('Data/kc_house_data.csv', target_dist)
        #self.data = RedfinData('Data/redfin_encoded.csv', target_dist)
        print('{}'.format(self.data.get_description()))
        
        print('Building neural network.')
        input_shape = self.data.X_train.shape[1:]
        num_outputs = self.data.num_targets
        self.network = NeuralNetwork(input_shape, num_outputs)
        
    def run(self):
        print('Training neural network.')
        self.network.train(self.data, 5)
        
        print('Evaluating neural network.')
        result = self.network.test(self.data)
    
        print('')
        print('Accuracy: {:.2f}%'.format(result*100))
        print('Done.')

def main(argv):
    app = Application()
    app.run()

if __name__ == '__main__':
    main(sys.argv[1:])
