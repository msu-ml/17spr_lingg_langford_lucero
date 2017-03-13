# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:24:58 2017

@author: Michael Austin Langford
"""

import csv
import numpy
import os
import re
from keras.utils import np_utils

class ReplaceMethod(object):
    MEAN = 1
    SIMILARITY = 2

class HousingData(object):
    def __init__(self, filepath, num_classes):
        data = self.read_csv(filepath)
        numpy.random.shuffle(data)

        (X, y) = self.preprocess_data(data)
        X = self.replace_missing_values(X, ReplaceMethod.SIMILARITY)

        self.num_classes = num_classes
        self.target_dist = self.get_target_distribution(y)
        y = self.categorize_targets(y)
        
        test_size = numpy.int(X.shape[0] * 0.1)
        self.X_train = X[0:-test_size]
        self.y_train = y[0:-test_size]
        self.X_test = X[-test_size:]
        self.y_test = y[-test_size:]
    
    def get_description(self):
        return ''
        
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
        return (None, None)
    
    def replace_missing_values(self, data, method):
        has_nan = numpy.isnan(data).any(axis=1)
        complete_data = data[~has_nan]
        if method == ReplaceMethod.MEAN:
            mean = numpy.mean(complete_data, axis=0)
            for entry in data:
                nan_idx = numpy.isnan(entry)
                if nan_idx.any():
                    entry[nan_idx] = mean[nan_idx]
        elif method == ReplaceMethod.SIMILARITY:
            for entry in data:
                nan_idx = numpy.isnan(entry)
                if nan_idx.any():
                    dist = numpy.sum((complete_data[:,~nan_idx] - entry[~nan_idx])**2, axis=1)
                    closest = numpy.argmin(dist)
                    entry[nan_idx] = complete_data[closest,nan_idx]  
        return data
    
    def get_target_distribution(self, targets):
        target_dist = []
        partition_size = numpy.int(targets.shape[0]/(self.num_classes-1))
        partitions = self.partition(numpy.sort(targets), partition_size)
        for partition in partitions:
            target_dist.append(partition[0])
        return target_dist

    def categorize_targets(self, targets):
        for i in range(len(targets)):
            target_class = 0
            for j in range(len(self.target_dist)):
                if targets[i] > self.target_dist[j]:
                    target_class = j 
            targets[i] = target_class
        return np_utils.to_categorical(targets, self.num_classes)

    def partition(self, data, n):
        for i in range(0, len(data), n):
            yield data[i:i + n]

class RedfinData(HousingData):
    def __init__(self, filepath, num_classes):
        super(RedfinData, self).__init__(filepath, num_classes)
        
    def get_description(self):
        return ('Housing data for the Grand Rapids, MI area.' + os.linesep +
                'Training data entries: {}'.format(self.X_train.shape[0]) + os.linesep +
                'Test data entries: {}'.format(self.X_test.shape[0]) + os.linesep +
                'Target classes: {}'.format(self.target_dist))
    
    def preprocess_data(self, data):
        X = numpy.copy(data)
        y = numpy.copy(data[:,14])
        
        # Remove columns.
        X = numpy.delete(X, [14], axis=1)
        
        # Convert all fields into appropriate data types
        X = X.astype('float32')
        y = y.astype('int')

        return (X, y)

class KingCountyData(HousingData):
    def __init__(self, filepath, num_classes):
        super(KingCountyData, self).__init__(filepath, num_classes)
        
    def get_description(self):
        return ('Housing data for the King County, WA area.' + os.linesep +
                'Training data entries: {}'.format(self.X_train.shape[0]) + os.linesep +
                'Test data entries: {}'.format(self.X_test.shape[0]) + os.linesep +
                'Target classes: {}'.format(self.target_dist))

    def preprocess_data(self, data):
        X = numpy.copy(data)
        y = numpy.copy(data[:,2])
        
        # Remove columns.
        X = numpy.delete(X, [0,1,2], axis=1)
        
        # Convert all fields into appropriate data types
        X = X.astype('float32')
        y = y.astype('float32')
        y = y.astype('int')

        return (X, y)

class NashvilleData(HousingData):
    def __init__(self, filepath, num_classes):
        super(NashvilleData, self).__init__(filepath, num_classes)

    def get_description(self):
        return ('Housing data for the Nashville, TN area.' + os.linesep +
                'Training data entries: {}'.format(self.X_train.shape[0]) + os.linesep +
                'Test data entries: {}'.format(self.X_test.shape[0]) + os.linesep +
                'Target classes: {}'.format(self.target_dist))

    def preprocess_data(self, data):
        X = numpy.copy(data)
        y = numpy.copy(data[:,8])
        
        landuse_column = 3
        landuse_encodings = {}
        landuse_regex = re.compile(r"""\s*(?P<value>.+)\s*""")     
        X = self.encode(X, landuse_column, landuse_encodings, landuse_regex)

        addr_column = 4
        addr_encodings = {}
        addr_regex = re.compile(r"""\s*(\d|\-|\&|\s)*\s*(?P<value>(\w\s*)+)\s*(#\d+)?\s*""")        
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
        X = numpy.delete(X, [0,1,2,5,8,9,12,13,14,15,19,31,32,33], axis=1)
        
        # Convert all fields into float values
        X = X.astype('float32')
        y = y.astype('int')

        return (X, y)
    
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
    