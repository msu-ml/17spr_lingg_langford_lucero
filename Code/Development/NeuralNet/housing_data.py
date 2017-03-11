# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:24:58 2017

@author: Michael Austin Langford
"""

import csv
import numpy
import re
from keras.utils import np_utils

class RedfinData(object):
    def __init__(self, filepath):
        self.num_targets = 10
        self.target_increments = 20000
        
        data = self.read_csv(filepath)
        numpy.random.shuffle(data)
        
        (X, y) = self.preprocess_data(data)
        y = self.preprocess_targets(y)
        
        test_size = numpy.int(X.shape[0] * 0.1)
        self.X_train = X[0:-test_size]
        self.y_train = y[0:-test_size]
        self.X_test = X[-test_size:]
        self.y_test = y[-test_size:]
        
    def read_csv(self, filepath):
        data = []
        with open(filepath, 'rb') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                entry = []
                for field in row:
                    if field != '':
                        entry.append(field)
                    else:
                        #entry.append(numpy.nan)
                        entry.append(0)
                data.append(entry)
        return numpy.array(data[1:])
    
    def preprocess_data(self, data):
        X = numpy.copy(data)
        y = numpy.copy(data[:,14])
        
        # Remove columns.
        X = numpy.delete(X, [14], axis=1)
        
        # Convert all fields into appropriate data types
        X = X.astype('float32')
        y = y.astype('int')

        return (X, y)
    
    def preprocess_targets(self, targets):
        for i in range(len(targets)):
            target_class = numpy.int(targets[i] / self.target_increments)
            target_class = min(target_class, self.num_targets - 1)
            targets[i] = target_class
        targets = np_utils.to_categorical(targets, self.num_targets)
        return targets

class KingCountyData(object):
    def __init__(self, filepath):
        self.num_targets = 15
        self.target_increments = 50000
        
        data = self.read_csv(filepath)
        (X, y) = self.preprocess_data(data)
        y = self.preprocess_targets(y)
        
        self.X_train = X[0:-2000]
        self.y_train = y[0:-2000]
        
        self.X_test = X[-2000:]
        self.y_test = y[-2000:]
        
    def read_csv(self, filepath):
        data = []
        with open(filepath, 'rb') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                entry = []
                for field in row:
                    if field != '':
                        entry.append(field)
                    else:
                        entry.append(numpy.nan)
                data.append(entry)
        return numpy.array(data[1:])
    
    def preprocess_data(self, data):
        X = numpy.copy(data)
        y = numpy.copy(data[:,2])
        
        # Remove columns.
        X = numpy.delete(X, [0,1,2], axis=1)
        
        # Convert all fields into appropriate data types
        X = X.astype('float32')
        y = y.astype('float32')

        return (X, y)
    
    def preprocess_targets(self, targets):
        for i in range(len(targets)):
            target_class = numpy.int(targets[i] / self.target_increments)
            target_class = min(target_class, self.num_targets - 1)
            targets[i] = target_class
        targets = np_utils.to_categorical(targets, self.num_targets)
        return targets

class NashvilleData(object):
    def __init__(self, filepath):
        self.num_targets = 15
        self.target_increments = 50000
        
        data = self.read_csv(filepath)
        (X, y) = self.preprocess_data(data)
        y = self.preprocess_targets(y)
        
        self.X_train = X[0:-2000]
        self.y_train = y[0:-2000]
        
        self.X_test = X[-2000:]
        self.y_test = y[-2000:]
        
    def read_csv(self, filepath):
        data = []
        with open(filepath, 'rb') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                entry = []
                for field in row:
                    if field != '':
                        entry.append(field)
                    else:
                        entry.append(numpy.nan)
                data.append(entry)
        return numpy.array(data[1:])
    
    def preprocess_data(self, data):
        X = numpy.copy(data)
        y = numpy.copy(data[:,8])
        
        landuse_column = 3
        landuse_encodings = {}
        landuse_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, landuse_column, landuse_encodings, landuse_regex)

        addr_column = 4
        addr_encodings = {}
        addr_regex = re.compile(r"""\s*\d*\s*(?P<value>.+)\s*""")        
        X = self.encode(X, addr_column, addr_encodings, addr_regex)
        
        city_column = 6
        city_encodings = {}
        city_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, city_column, city_encodings, city_regex)

        date_column = 7
        date_encodings = {}
        date_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, date_column, date_encodings, date_regex)

        vacant_column = 10
        vacant_encodings = {}
        vacant_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, vacant_column, vacant_encodings, vacant_regex)

        parcels_column = 11
        parcels_encodings = {}
        parcels_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, parcels_column, parcels_encodings, parcels_regex)

        district_column = 17
        district_encodings = {}
        district_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, district_column, district_encodings, district_regex)

        foundation_column = 24
        foundation_encodings = {}
        foundation_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, foundation_column, foundation_encodings, foundation_regex)

        exterior_column = 26
        exterior_encodings = {}
        exterior_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, exterior_column, exterior_encodings, exterior_regex)

        grade_column = 27
        grade_encodings = {}
        grade_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, grade_column, grade_encodings, grade_regex)

        # Remove unnecessary columns.
        X = numpy.delete(X, [0,1,2,5,8,9,12,13,14,15,19], axis=1)
        
        # Convert all fields into float values
        X = X.astype('float32')
        y = y.astype('int')

        # Remove any entries with empty fields
        missing_values = numpy.isnan(X).any(axis=1)
        X = X[~missing_values]
        y = y[~missing_values]

        return (X, y)
    
    def preprocess_targets(self, targets):
        for i in range(len(targets)):
            target_class = numpy.int(targets[i] / self.target_increments)
            target_class = min(target_class, self.num_targets - 1)
            targets[i] = target_class
        targets = np_utils.to_categorical(targets, self.num_targets)
        return targets
    
    def encode(self, data, column, encodings, regex):
        for entry in data:
            match = regex.match(entry[column])
            if match != None:
                value = match.group('value')
                if not value in encodings.keys():
                    encodings[value] = len(encodings) + 1
                entry[column] = encodings[value]
            else:
                entry[column] = numpy.nan
        return data