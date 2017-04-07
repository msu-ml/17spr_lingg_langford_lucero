# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:24:58 2017

@author: Michael Austin Langford
"""

import csv
import numpy
import os
import sys

class SubstitutionMethod(object):
    NONE = 0
    MEAN = 1
    CLOSEST_VALUE = 2
    CLOSEST_MEAN = 3

class HousingData(object):
    def __init__(self,
                 filepath,
                 name='Unnamed',
                 preprocessed=True,
                 fields=[],
                 target_field=None,
                 cat_fields=[],
                 empty_value='',
                 subMethod=SubstitutionMethod.CLOSEST_MEAN,
                 normalize=True):
        if preprocessed:
            # Read data from preprocessed csv file.
            data, fields = self.read_processed_csv(filepath)

            # Read unnormalized data bounds from preprocssed csv file.
            bounds_filepath = os.path.splitext(filepath)[0] + '_bounds.csv'
            bounds, _ = self.read_processed_csv(bounds_filepath)
            self.data_min = (bounds[0,:-1], bounds[0,-1])
            self.data_max = (bounds[1,:-1], bounds[1,-1])
            
            # Separate the target field from the rest.
            (X, y), fields = self.separate_targets(data, fields, fields[-1])
        else:
            # Read data from a csv file.
            data, fields = self.read_unprocessed_csv(filepath, fields, cat_fields, empty_value)
        
            # Separate the target field from the rest.
            if target_field == None:
                target_field = fields[-1]
            (X, y), fields = self.separate_targets(data, fields, target_field)
        
            # Replace any missing values with a substitute.
            if subMethod != SubstitutionMethod.NONE:
                X = self.replace_missing_values(X, subMethod)

            # Normalize values by column.
            if normalize:
                X, X_min, X_max = self.normalize_values(X)
                y, y_min, y_max = self.normalize_values(y)
                self.data_min = (X_min, y_min)
                self.data_max = (X_max, y_max)
            
        # reshape data
        if X.ndim > 1:
            X = [numpy.reshape(x, (X.shape[1], 1)) for x in X]
        else:
            X = [numpy.array(x, ndmin=2, copy=False) for x in X]
        if y.ndim > 1:
            y = [numpy.reshape(t, (y.shape[1], 1)) for t in y]
        else:
            y = [numpy.array(t, ndmin=2, copy=False) for t in y]

        self.name = name
        self.fields = fields
        self.data = [(X[i], y[i]) for i in xrange(len(X))]
        self.num_features = X[0].shape[0]        
       
    def get_name(self):
        return self.name
    
    def get_num_entries(self):
        return len(self.data)
    
    def get_num_features(self):
        return self.num_features
    
    def split_data(self, a, b):
        test_size = int(len(self.data) * float(b) / float(a + b))
        data_train = self.data[0:-test_size]
        data_test = self.data[-test_size:]
        return data_train, data_test
    
    def create_classes(self, num_classes):
        classes = []
        targets = zip(*self.data)[1]
        batch_size = numpy.int(len(targets) / num_classes)
        batches = self.make_batches(sorted(targets), batch_size)
        for batch in batches:
            classes.append(batch[0])
        if len(classes) > num_classes:
            classes = [classes[i] for i in xrange(num_classes)]
        return classes

    def classify_targets(self, data, classes):
        temp = zip(*data)
        X = temp[0]
        y = [self.classify_target(y, classes) for y in temp[1]]
        data = [(X[i], y[i]) for i in xrange(len(data))]
        return data
    
    def classify_target(self, target, classes):
        target_class = 0
        for j in xrange(len(classes)):
            if target > classes[j]:
                target_class = j
        t = numpy.zeros((len(classes), 1))
        t[target_class] = 1.0
        return t
    
    def make_batches(self, data, batch_size):
        for i in xrange(0, len(data), batch_size):
            yield data[i:i+batch_size]
    
    def read_processed_csv(self, data_filepath):
        data = []
        with open(data_filepath, 'rb') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                data.append(row)
                
        fields = data[0]
        data = numpy.asarray(data[1:], dtype=numpy.float32)
        
        return data, fields
        
    def read_unprocessed_csv(self, filepath, fields, cat_fields, empty_value):
        data = []
        
        # Open file for reading.
        with open(filepath, 'rb') as input_file:
            # Use a CSV dictionary reader to read each desired field
            reader = csv.DictReader(input_file)
            for row in reader:
                entry = []
                for field in fields:
                    # If the field is empty, put a nan value in it.
                    if row[field] != '' and row[field] != empty_value:
                        entry.append(row[field])
                    else:
                        entry.append(numpy.nan)
                data.append(entry)
        data = numpy.asarray(data)
        
        # Split categorical fields up.
        if len(cat_fields) > 0:
            data, fields = self.split_categorical_fields(data, fields, cat_fields)
        data = numpy.asarray(data, dtype=numpy.float32)
        
        return data, fields
    
    def split_categorical_fields(self, data, fields, cat_fields):
        for field in cat_fields:
            column = fields.index(field)
            cats = numpy.unique(data[:,column])
            cats = numpy.delete(cats, numpy.argwhere(cats=='nan'))
            cat_data = numpy.zeros((data.shape[0], cats.shape[0]))
            for i in xrange(cats.shape[0]):
                cat_data[:,i] = (data[:,column] == cats[i])
                new_field = field + '_is_' + cats[i].strip().title()
                fields.append(new_field)
            data = numpy.concatenate((data, cat_data), axis=1)
        
        delete_columns = []
        for field in cat_fields:
            delete_columns.append(fields.index(field))
        data = numpy.delete(data, delete_columns, axis=1)
        fields = [f for f in fields if f not in cat_fields]
        
        return data, fields
    
    def replace_missing_values(self, data, method):
        # Determine where all the nan values are in the data array.
        nan_idx = numpy.isnan(data)
        
        # Determine which rows have nan values.
        nan_rows = numpy.any(nan_idx, axis=1)
        
        # Get only the rows with no nan values.
        complete_data = data[~nan_rows]
        
        if method == SubstitutionMethod.MEAN:
            # Determine the mean value for each column.
            mean = numpy.mean(complete_data, axis=0)
            
            # For each entry, replace its nan values with the column's mean.
            for entry in data:
                nan_idx = numpy.isnan(entry)
                if nan_idx.any():
                    entry[nan_idx] = mean[nan_idx]
        elif method == SubstitutionMethod.CLOSEST_VALUE:
            for entry in data:
                # Find the nan values in the current entry.
                nan_idx = numpy.isnan(entry)
                if nan_idx.any():
                    # Compare the non-nan values of this entry with every other entry.
                    dist = numpy.sum((complete_data[:,~nan_idx] - entry[~nan_idx])**2, axis=1)
                    
                    # Get the index of the entry that is most similar.
                    closest_idx = numpy.argmin(dist)
                    
                    # Replace the nan values in this entry with the corresponding
                    # values in the closest entry.
                    entry[nan_idx] = complete_data[closest_idx,nan_idx]
        elif method == SubstitutionMethod.CLOSEST_MEAN:
            for entry in data:
                # Find the nan values in the current entry.
                nan_idx = numpy.isnan(entry)
                if nan_idx.any():
                    # Compare the non-nan values of this entry with every other entry.
                    dist = numpy.sum((complete_data[:,~nan_idx] - entry[~nan_idx])**2, axis=1)
                    
                    # Get the index of the ten closets entries in value to the
                    # current entry
                    closest_idx = numpy.argsort(dist)[0:10]
                    
                    # Replace the nan values in this entry with the corresponding
                    # mean from the ten closest entries.
                    mean = numpy.mean(complete_data[closest_idx], axis=0)
                    entry[nan_idx] = mean[nan_idx]
                    
        return data
    
    def normalize_values(self, data):
        max_vals = numpy.amax(data, axis=0)
        min_vals = numpy.amin(data, axis=0)
        data = (data - min_vals) / (max_vals - min_vals)
        return data, min_vals, max_vals

    def unnormalize_target(self, value):
        y_max = self.data_max[1]
        y_min = self.data_min[1]
        value = ((y_max - y_min) * value) + y_min
        return value
        
    def separate_targets(self, data, fields, target_field):
        target_column = fields.index(target_field)

        X = numpy.copy(data)
        y = numpy.copy(data[:,target_column])
        X = numpy.delete(X, [target_column], axis=1)

        X_fields = [f for f in fields if f != target_field]
        y_fields = [target_field]
        return (X, y), (X_fields, y_fields)
    
    def write_csv(self, filepath):
        # Write processed data to csv file.
        with open(filepath, 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(numpy.hstack(self.fields))
            for X, y in self.data:
                writer.writerow(numpy.hstack(numpy.vstack((X, y))))
        
        # Write boundary values from normalization to csv file.
        bounds_filepath = os.path.splitext(filepath)[0] + '_bounds.csv'
        with open(bounds_filepath, 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(numpy.hstack(self.fields))
            writer.writerow(numpy.hstack(self.data_min))
            writer.writerow(numpy.hstack(self.data_max))