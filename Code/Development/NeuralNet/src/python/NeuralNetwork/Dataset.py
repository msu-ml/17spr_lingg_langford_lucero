# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import numpy as np
import sys

class Dataset(object):
    def __init__(self, data, targets):
        self.__num_entries = data.shape[0]
        self.__num_features = data.shape[1]
        self.__dataset = np.hstack((data, targets))

    def get_data(self):
        return self.__dataset[:,:self.num_features]
    def set_data(self, v):
        self.__dataset[:,:self.num_features] = v
    data = property(fget=lambda self: self.get_data(),
                    fset=lambda self, v: self.set_data(v))
    
    def get_targets(self):
        return self.__dataset[:,self.num_features:]
    def set_targets(self, v):
        self.__dataset[:,self.num_features:] = v
    targets = property(fget=lambda self: self.get_targets(),
                       fset=lambda self, v: self.set_targets(v))
        
    def get_num_entries(self):
        return self.__num_entries
    num_entries = property(fget=lambda self: self.get_num_entries())
    
    def get_num_features(self):
        return self.__num_features
    num_features = property(fget=lambda self: self.get_num_features())

    def get_num_targets(self):
        return self.__dataset.shape[1] - self.__num_features
    num_targets = property(fget=lambda self: self.get_num_targets())

    def shuffle(self):
        np.random.shuffle(self.__dataset)

    def split(self, ratio):
        n = int(self.num_entries * ratio)
        dataset1 = Dataset(self.data[:n,:], self.targets[:n,:])
        dataset2 = Dataset(self.data[n:,:], self.targets[n:,:])
        return dataset1, dataset2

    def create_classes(self, num_classes):
        classes = np.zeros(num_classes)
        targets = np.sort(np.copy(self.targets), axis=0)
        batch_size = np.int(len(targets) / num_classes)
        num_targets = batch_size * num_classes
        j = 0
        for i in xrange(0, num_targets, batch_size):
            batch = targets[i:i+batch_size]
            classes[j] = batch[0]
            j += 1
        return classes

    def encode_targets(self, classes):
        targets = self.targets
        enc_targets = np.zeros((self.num_entries, classes.shape[0]))
        for i in xrange(self.num_entries):
            enc_targets[i,:] = self.encode_target(targets[i][0], classes)
        self.__dataset = np.hstack((self.data, enc_targets))
    
    def encode_target(self, target, classes):
        target_class = 0
        for j in xrange(len(classes)):
            if target >= classes[j]:
                target_class = j
        t = np.zeros((1, classes.shape[0]))
        t[0, target_class] = 1.0
        return t
    
    def make_batches(self, batch_size):
        batches = []
        for i in xrange(0, self.num_entries, batch_size):
            batch_data = self.data[i:i+batch_size,:]
            batch_targets = self.targets[i:i+batch_size,:]
            batches.append(Dataset(batch_data, batch_targets))
        return batches