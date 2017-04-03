# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import csv
import getopt
import numpy
import os
import re
import sys

seed = 69
numpy.random.seed(seed)

class SubstitutionMethod(object):
    NONE = 0
    MEAN = 1
    CLOSEST_NEIGHBOR_VALUE = 2
    CLOSEST_NEIGHBOR_MEAN = 3
    
class Options():
    def __init__(self):
        self.data_filepath = ''
        self.field_filepath = ''
        self.output_filepath = ''
        self.empty_value = ''
        self.random_shuffle = False
        self.normalize = False
        self.substitution = SubstitutionMethod.NONE

class Data(object):
    def __init__(self, opts):
        """Constructs a new data object.
        """
        self.opts = opts
        (self.fields, self.cat_fields) = self.read_fields(self.opts.field_filepath)
        self.data = self.read_csv(self.opts.data_filepath)
        self.processed_data = numpy.array([]);
    
    def read_fields(self, field_filepath):
        fields = []
        cat_fields = []
        
        with open(self.opts.field_filepath) as input_file:
            lines = input_file.readlines()
    
        field_regex = re.compile(r"^\s*[\+\*\s]+\s*(?P<field>[\w\s]+)\s*\n$", re.MULTILINE)
        cat_field_regex = re.compile(r"^\s*[\*]+\s*(?P<cat_field>[\w\s]+)\s*\n$", re.MULTILINE)
        for line in lines:
            match = field_regex.match(line)
            if (match):
                fields.append(match.group('field'))
            match = cat_field_regex.match(line)
            if (match):
                cat_fields.append(match.group('cat_field'))
        
        return fields, cat_fields
        
    def read_csv(self, filepath):
        """Reads a CSV file.
        Arguments:
            filepath - the file path to a CSV file
        """
        data = []

        with open(filepath, 'rb') as input_file:
            # Use a CSV dictionary reader to read each desired field
            reader = csv.DictReader(input_file)
            for row in reader:
                entry = []
                for field in self.fields:
                    # If the field is empty, put a nan value in it.
                    if row[field] != self.opts.empty_value and row[field] != '':
                        entry.append(row[field])
                    else:
                        entry.append(numpy.nan)
                data.append(entry)
                
        return numpy.array(data)
    
    def process(self):
        self.processed_data = numpy.copy(self.data)
        
         # Randomly shuffle the data rows.
        if self.opts.random_shuffle:
            numpy.random.shuffle(self.processed_data)

        # Split categorical fields up.
        if len(self.cat_fields) > 0:
            self.processed_data = self.split_categorical_fields(self.processed_data, self.cat_fields)
        self.processed_data = self.processed_data.astype('float32')
        
        # Replace any missing values with a substitute.
        if self.opts.substitution != SubstitutionMethod.NONE:
            self.processed_data = self.replace_missing_values(
                    self.processed_data,
                    self.opts.substitution)
        
        # Normalize values by column
        if self.opts.normalize == True:
            self.processed_data, self.max_vals, self.min_vals = self.normalize_values(self.processed_data)
    
    def write_csv(self):
        if self.opts.output_filepath != '':
            output_path = self.opts.output_filepath
        else:
            output_path = os.path.splitext(self.opts.data_filepath)[0] + '_processed.csv'
        output_fields = numpy.array(self.fields)
        output_data = numpy.vstack([output_fields, self.processed_data])
        with open(output_path, 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(output_data)
    
    def split_categorical_fields(self, data, cat_fields):
        """Finds unique values for a categorical field and splits it into a set
        of binary fields for each category.
        Arguments:
            data - a data array
            cat_fields - the fields that contain categorical values
        """
        for field in cat_fields:
            column = self.fields.index(field)
            cats = numpy.unique(data[:,column])
            cats = numpy.delete(cats, numpy.argwhere(cats=='nan'))
            cat_data = numpy.zeros((data.shape[0], cats.shape[0]))
            for i in range(cats.shape[0]):
                cat_data[:,i] = (data[:,column] == cats[i])
                new_field = field + ' (is ' + cats[i].strip().title() + ')'
                self.fields.append(new_field)
            data = numpy.concatenate((data, cat_data), axis=1)
        
        delete_columns = []
        for field in cat_fields:
            delete_columns.append(self.fields.index(field))
        data = numpy.delete(data, delete_columns, axis=1)
        self.fields = [x for x in self.fields if x not in cat_fields]
        
        return data
    
    def replace_missing_values(self, data, method):
        """Replaces any missing data with a substitute.
        Arguments:
            data - an array of data
            method - the substitution method
        """
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
        elif method == SubstitutionMethod.CLOSEST_NEIGHBOR_VALUE:
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
        elif method == SubstitutionMethod.CLOSEST_NEIGHBOR_MEAN:
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
        """Normalizes the values in the data by dividing by the column's max.
        Arguments:
            data - an array of data
        """
        max_vals = numpy.amax(data, axis=0)
        min_vals = numpy.amin(data, axis=0)
        return (data - min_vals) / (max_vals - min_vals), max_vals, min_vals

def main(argv):
    try:                                
        opts, args = getopt.getopt(argv,
                                   'f:o:x:rns:',
                                   ['fields', 'output', 'missing', 'random',
                                    'normalize', 'sub'])
    except getopt.GetoptError:
        sys.exit(2)

    options = Options()
    for opt, arg in opts:
        if opt in ('-f', '--fields'):
            options.field_filepath = arg
        elif opt in ('-o', '--output'):
            options.output_filepath = arg
        elif opt in ('-x', '--missing'):
            options.empty_value = arg
        elif opt in ('-r', '--random'):
            options.random_shuffle = True
        elif opt in ('-n', '--normalize'):
            options.normalize = True
        elif opt in ('-s', '-sub'):
            method = arg.lower()
            if method == 'mean':
                options.substitution = SubstitutionMethod.MEAN
            elif method == 'closest_value':
                options.substitution = SubstitutionMethod.CLOSEST_NEIGHBOR_VALUE
            elif method == 'closest_mean':
                options.substitution = SubstitutionMethod.CLOSEST_NEIGHBOR_MEAN

    if len(args) > 0:
        options.data_filepath = args[0]
    
        print('Loading data...')
        data = Data(options)
        
        print('Processing data...')
        data.process()
        
        print('Writing processed data...')
        data.write_csv()

if __name__ == '__main__':
    main(sys.argv[1:])
