# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:24:58 2017

@author: Michael Austin Langford
"""

import csv
import numpy as np
import os
import string
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
                 normalize=True,
                 target_bounds=None):
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
            given_target_field = target_field
            
            # Read data from a csv file.
            if given_target_field == None:
                target_field = fields[-1]
            data, fields = self.read_unprocessed_csv(filepath, fields, cat_fields, target_field, target_bounds, empty_value)
            
            # Separate the target field from the rest.
            if given_target_field == None:
                target_field = fields[-1]
            (X, y), fields = self.separate_targets(data, fields, target_field)
            
            # Replace any missing values with a substitute.
            if subMethod != SubstitutionMethod.NONE:
                X = self.replace_missing_values(X, subMethod)
            
            # Normalize values by column.
            if normalize:
                X, X_min, X_max = self.normalize_values(X)
                y, y_min, y_max = self.normalize_values(y, bounds=target_bounds)
                self.data_min = (X_min, y_min)
                self.data_max = (X_max, y_max)
                
        if y.ndim == 1:
            y = np.reshape(y, (y.shape[0], 1))

        self.name = name
        self.fields = fields
        self.data = (X, y)

    def get_name(self):
        """Gets the name of the data set.
        """
        return self.__name
    def set_name(self, v):
        """Sets the name of the data set.
        """
        self.__name = v
    name = property(fget=lambda self: self.get_name(),
                    fset=lambda self, v: self.set_name(v))

    def get_fields(self):
        """Gets the data fields.
        """
        return self.__fields
    def set_fields(self, v):
        """Sets the data fields.
        """
        self.__fields = v
    fields = property(fget=lambda self: self.get_fields(),
                      fset=lambda self, v: self.set_fields(v))

    def get_data(self):
        """Gets the data set.
        """
        return self.__data
    def set_data(self, v):
        """Sets the data set.
        """
        self.__data = v
    data = property(fget=lambda self: self.get_data(),
                    fset=lambda self, v: self.set_data(v))
    
    def get_data_max(self):
        """Gets the max values of the unnormalized data.
        """
        return self.__data_max
    def set_data_max(self, v):
        """Sets the max values of the unnormalized data
        """
        self.__data_max = v
    data_max = property(fget=lambda self: self.get_data_max(),
                        fset=lambda self, v: self.set_data_max(v))

    def get_data_min(self):
        """Gets the min values of the unnormalized data.
        """
        return self.__data_min
    def set_data_min(self, v):
        """Sets the min values of the unnormalized data
        """
        self.__data_min = v
    data_min = property(fget=lambda self: self.get_data_min(),
                        fset=lambda self, v: self.set_data_min(v))
    
    def read_processed_csv(self, data_filepath):
        data = []
        with open(data_filepath, 'rb') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                data.append(row)
                
        fields = data[0]
        data = np.asarray(data[1:], dtype=np.float32)
        
        return data, fields
        
    def read_unprocessed_csv(self, filepath, fields, cat_fields, target_field, target_bounds, empty_value):
        data = []
        
        if not target_bounds is None:
            target_min, target_max = target_bounds
        
        # Open file for reading.
        with open(filepath, 'rb') as input_file:
            # Use a CSV dictionary reader to read each desired field
            reader = csv.DictReader(input_file)
            for row in reader:
                entry = []
                target_value = np.nan
                for field in fields:
                    # If the field is empty, put a nan value in it.
                    if row[field] != '' and row[field] != empty_value:
                        entry.append(row[field])
                    else:
                        entry.append(np.nan)
                    if field.upper().strip() == target_field.upper().strip():
                        target_value = np.float(row[field])
                # only include entries that fall within the target_bounds
                if not target_bounds is None:
                    if target_value >= target_min and target_value <= target_max:
                        data.append(entry)
                else:
                    data.append(entry)
        data = np.asarray(data)
        
        # Split categorical fields up.
        if len(cat_fields) > 0:
            data, fields = self.split_categorical_fields(data, fields, cat_fields)
        fields = [f.upper().replace(' ', '_').replace('/', '_').replace(':', '_').replace('\\', '_').replace('(', '').replace(')', '').strip() for f in fields]
        data = np.asarray(data, dtype=np.float32)
        
        return data, fields
    
    def split_categorical_fields(self, data, fields, cat_fields):
        for field in cat_fields:
            column = fields.index(field)
            cats = np.unique(data[:,column])
            cats = np.delete(cats, np.argwhere(cats=='nan'))
            #cat_data = np.zeros((data.shape[0], cats.shape[0]))
            for i in xrange(cats.shape[0]):
                cat_data = (data[:,column] == cats[i]).astype(np.float32)
                new_field = field + ' is ' + cats[i]
                fields.insert(-1, new_field)
                data = np.insert(data, -1, cat_data, axis=1)
        
        delete_columns = []
        for field in cat_fields:
            delete_columns.append(fields.index(field))
        data = np.delete(data, delete_columns, axis=1)
        fields = [f for f in fields if f not in cat_fields]
        
        return data, fields
    
    def replace_missing_values(self, data, method):
        # Determine where all the nan values are in the data array.
        nan_idx = np.isnan(data)
        
        # Determine which rows have nan values.
        nan_rows = np.any(nan_idx, axis=1)
        
        # Get only the rows with no nan values.
        complete_data = data[~nan_rows]
        
        if method == SubstitutionMethod.MEAN:
            # Determine the mean value for each column.
            mean = np.mean(complete_data, axis=0)
            
            # For each entry, replace its nan values with the column's mean.
            for entry in data:
                nan_idx = np.isnan(entry)
                if np.any(nan_idx):
                    entry[nan_idx] = mean[nan_idx]
        elif method == SubstitutionMethod.CLOSEST_VALUE:
            for entry in data:
                # Find the nan values in the current entry.
                nan_idx = np.isnan(entry)
                if nan_idx.any():
                    # Compare the non-nan values of this entry with every other entry.
                    dist = np.sum((complete_data[:,~nan_idx] - entry[~nan_idx])**2, axis=1)
                    
                    # Get the index of the entry that is most similar.
                    closest_idx = np.argmin(dist)
                    
                    # Replace the nan values in this entry with the corresponding
                    # values in the closest entry.
                    entry[nan_idx] = complete_data[closest_idx,nan_idx]
        elif method == SubstitutionMethod.CLOSEST_MEAN:
            for entry in data:
                # Find the nan values in the current entry.
                nan_idx = np.isnan(entry)
                if nan_idx.any():
                    # Compare the non-nan values of this entry with every other entry.
                    dist = np.sum((complete_data[:,~nan_idx] - entry[~nan_idx])**2, axis=1)
                    
                    # Get the index of the ten closets entries in value to the
                    # current entry
                    closest_idx = np.argsort(dist)[0:10]
                    
                    # Replace the nan values in this entry with the corresponding
                    # mean from the ten closest entries.
                    mean = np.mean(complete_data[closest_idx], axis=0)
                    entry[nan_idx] = mean[nan_idx]
                    
        return data
    
    def normalize_values(self, data, bounds=None):
        if bounds is None:
            min_vals = np.amin(data, axis=0)
            max_vals = np.amax(data, axis=0)
        else:
            min_vals, max_vals = bounds
        data = (data - min_vals) / (max_vals - min_vals)
        return data, min_vals, max_vals

    def normalize_target(self, value):
        y_max = self.data_max[1]
        y_min = self.data_min[1]
        value = (value - y_min) / (y_max - y_min)
        return value

    def unnormalize_target(self, value):
        y_max = self.data_max[1]
        y_min = self.data_min[1]
        value = ((y_max - y_min) * value) + y_min
        return value
        
    def separate_targets(self, data, fields, target_field):
        target_column = fields.index(target_field)

        X = np.copy(data)
        y = np.copy(data[:,target_column])
        X = np.delete(X, [target_column], axis=1)

        X_fields = [f for f in fields if f != target_field]
        y_fields = [target_field]
        return (X, y), (X_fields, y_fields)
    
    def write_csv(self, filepath):
        # Write processed data to csv file.
        with open(filepath, 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(np.hstack(self.fields))
            data = np.hstack(self.data)
            for row in data:
                writer.writerow(row)
        
        # Write boundary values from normalization to csv file.
        bounds_filepath = os.path.splitext(filepath)[0] + '_bounds.csv'
        with open(bounds_filepath, 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerow(np.hstack(self.fields))
            writer.writerow(np.hstack(self.data_min))
            writer.writerow(np.hstack(self.data_max))