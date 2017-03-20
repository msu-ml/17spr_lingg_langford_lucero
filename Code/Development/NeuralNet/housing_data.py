# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:24:58 2017

@author: Michael Austin Langford
"""

import csv
import numpy
import os
from keras.utils import np_utils

class SubstitutionMethod(object):
    MEAN = 1
    SIMILARITY = 2

class HousingData(object):
    def __init__(self, filepath, fields, cat_fields, target_field, num_classes):
        """Constructs a new data object.
        Arguments:
            filepath - file path for a CSV file
            fields - the fields to read
            cat_fields - the fields that contain categorical values
            target_field - the field containing the target values
            num_classes - the number of desired target classes
        """
        self.fields = fields
        self.num_classes = num_classes

        self.max_values = []
        
        # Read data from a csv file.
        data = self.read_csv(filepath)
        
        # Randomly shuffle the data rows.
        numpy.random.shuffle(data)

        # Split categorical fields up.
        if len(cat_fields) > 0:
            data = self.split_categorical_fields(data, cat_fields)
            
            # Write a copy of the processed data to a csv file.
            output_path = os.path.splitext(filepath)[0] + '_processed.csv'
            self.write_csv(output_path, self.fields, data)
        
        # Separate the target field from the rest.
        data = data.astype('float32')
        (X, y) = self.separate_targets(data, target_field)
        X = X.astype('float32')
        y = y.astype('int')
        
        # Replace any missing values with a substitute.
        X = self.replace_missing_values(X, SubstitutionMethod.SIMILARITY)
        
        # Normalize values by column.
        X = self.normalize_values(X)        
        
        # Determine the target distribution and categorize the target classes.
        self.target_dist = self.get_target_distribution(y)
        y = self.categorize_targets(y)
        
        # Separate the training data from the test data.
        test_size = numpy.int(X.shape[0] * 0.1)
        self.X_train = X[0:-test_size]
        self.y_train = y[0:-test_size]
        self.X_test = X[-test_size:]
        self.y_test = y[-test_size:]
        
    def get_description(self):
        """Gets a description of the data.
        """
        return ''
    
    def read_csv(self, filepath):
        """Reads a CSV file.
        Arguments:
            filepath - the file path to a CSV file
        """
        data = []
        
        # Open file for reading.
        with open(filepath, 'rb') as input_file:
            # Use a CSV dictionary reader to read each desired field
            reader = csv.DictReader(input_file)
            for row in reader:
                entry = []
                for field in self.fields:
                    # If the field is empty, put a nan value in it.
                    if row[field] != '':
                        entry.append(row[field])
                    else:
                        entry.append(numpy.nan)
                data.append(entry)
        return numpy.array(data)
    
    def write_csv(self, filepath, fields, data):
        output_fields = numpy.array(fields)
        output_data = numpy.vstack([output_fields, data])
        with open(filepath, 'wb') as output_file:
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
        elif method == SubstitutionMethod.SIMILARITY:
            for entry in data:
                # Find the nan values in the current entry.
                nan_idx = numpy.isnan(entry)
                if nan_idx.any():
                    # Compare the non-nan values of this entry with every other entry.
                    dist = numpy.sum((complete_data[:,~nan_idx] - entry[~nan_idx])**2, axis=1)
                    
                    # Get the index of the entry that is most similar.
                    closest = numpy.argmin(dist)
                    
                    # Replace the nan values in this entry with the corresponding
                    # values in the closest entry.
                    entry[nan_idx] = complete_data[closest,nan_idx]  
        return data
    
    def normalize_values(self, data):
        """Normalizes the values in the data by dividing by the column's max.
        Arguments:
            data - an array of data
        """
        self.max_values = numpy.amax(numpy.absolute(data), axis=0)
        return data / self.max_values
        
    def separate_targets(self, data, field):
        """Separates the target values from the data set.
        Arguments:
            data - an array of data
            field - the field containing the target values
        """
        target_column = self.fields.index(field)
        y = numpy.copy(data[:,target_column])
        X = numpy.copy(data)
        X = numpy.delete(X, [target_column], axis=1)
        
        return (X, y)
    
    def get_target_distribution(self, targets):
        """Computes a target distribution that partitions the data into equal-
        sized classes.
        Arguments:
            targets - the target values
        """
        target_dist = []
        partition_size = numpy.int(targets.shape[0]/(self.num_classes-1))
        partitions = self.partition(numpy.sort(targets), partition_size)
        for partition in partitions:
            target_dist.append(partition[0])
        return target_dist

    def categorize_targets(self, targets):
        """Converts target values into categorical classes for the neural net.
        Arguments:
            targets - the target values
        """
        for i in range(len(targets)):
            # Determine which class the value falls into, according to the 
            # target distribution.
            target_class = 0
            for j in range(len(self.target_dist)):
                if targets[i] > self.target_dist[j]:
                    target_class = j 
            targets[i] = target_class
        return np_utils.to_categorical(targets, self.num_classes)

    def partition(self, data, n):
        """Partitions the given data set into n equally sized partitions.
        Arguments:
            data - an array of data
            n - the number of desired partitions
        """
        for i in range(0, len(data), n):
            yield data[i:i + n]

class RedfinData(HousingData):
    """Represents housing data from redfin.com
    """
    def __init__(self, filepath, num_classes):
        """Constructs a new Redfin housing data object.
        Arguments:
            filepath - the file path to the Redfin csv file
            num_clases - the number of desired target classes
        """
        fields = [
                'PROPERTY TYPE',
                'CITY',
                'STATE',
                'ZIP',
                'BEDS',
                'BATHS',
                'SQUARE FEET',
                'LOT SIZE',
                'YEAR BUILT',
                'DAYS ON MARKET',
                'HOA/MONTH',
                'LATITUDE',
                'LONGITUDE',
                'PRICE'
                ]
        cat_fields = [
                'PROPERTY TYPE',
                'CITY',
                'STATE'
                ]
        target_field = 'PRICE'
        super(RedfinData, self).__init__(
                filepath, 
                fields, 
                cat_fields,
                target_field,
                num_classes)
        
    def get_description(self):
        """Gets a description of the housing data.
        """
        return ('Housing data for the Grand Rapids, MI area.' + os.linesep +
                'Training data entries: {}'.format(self.X_train.shape[0]) + os.linesep +
                'Test data entries: {}'.format(self.X_test.shape[0]) + os.linesep +
                'Target classes: {}'.format(self.target_dist))

class KingCountyData(HousingData):
    """Represents King Country housing data
    """
    def __init__(self, filepath, num_classes):
        """Constructs a new King County housing data object.
        Arguments:
            filepath - the file path to the King County csv file
            num_clases - the number of desired target classes
        """
        fields = [
                'price',
                'bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated',
                'zipcode',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15'
                ]
        cat_fields = []
        target_field = 'price'
        super(KingCountyData, self).__init__(
                filepath, 
                fields, 
                cat_fields,
                target_field,
                num_classes)
        
    def get_description(self):
        """Gets a description of the housing data.
        """
        return ('Housing data for the King County, WA area.' + os.linesep +
                'Training data entries: {}'.format(self.X_train.shape[0]) + os.linesep +
                'Test data entries: {}'.format(self.X_test.shape[0]) + os.linesep +
                'Target classes: {}'.format(self.target_dist))

class NashvilleData(HousingData):
    """Represents Nashville housing data
    """
    def __init__(self, filepath, num_classes):
        """Constructs a new Nashville housing data object.
        Arguments:
            filepath - the file path to the Nashville csv file
            num_clases - the number of desired target classes
        """
        fields = [
                'Land Use',
                'Property City',
                'Sale Price',
                'Sold As Vacant',
                'Multiple Parcels Involved in Sale',
                'Acreage',
                'Tax District',
                'Neighborhood',
                'Land Value',
                'Building Value',
                'Total Value',
                'Finished Area',
                'Foundation Type',
                'Year Built',
                'Exterior Wall',
                'Grade',
                'Bedrooms',
                'Full Bath',
                'Half Bath',
                'geocoded_zipcode',
                'geocoded_latitude',
                'geocoded_longitude'
                ]
        cat_fields = [
                'Land Use',
                'Property City',
                'Sold As Vacant',
                'Multiple Parcels Involved in Sale',
                'Tax District',
                'Foundation Type',
                'Exterior Wall',
                'Grade'
                ]
        target_field = 'Sale Price'
        super(NashvilleData, self).__init__(
                filepath,
                fields,
                cat_fields,
                target_field,
                num_classes)

    def get_description(self):
        """Gets a description of the housing data.
        """
        return ('Housing data for the Nashville, TN area.' + os.linesep +
                'Training data entries: {}'.format(self.X_train.shape[0]) + os.linesep +
                'Test data entries: {}'.format(self.X_test.shape[0]) + os.linesep +
                'Target classes: {}'.format(self.target_dist))