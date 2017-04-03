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
    MEAN = 1
    CLOSEST_NEIGHBOR_VALUE = 2
    CLOSEST_NEIGHBOR_MEAN = 3

class HousingData(object):
    def __init__(self,
                 filepath,
                 fields,
                 target_field,
                 num_classes,
                 cat_fields=[],
                 empty_value=''):
        self.fields = fields
        self.num_classes = num_classes

        # Read data from a csv file.
        data = self.read_csv(filepath, empty_value)
        
        # Split categorical fields up.
        if len(cat_fields) > 0:
            data = self.split_categorical_fields(data, cat_fields)
        data = numpy.asarray(data, dtype=numpy.float32)
        
        # Separate the target field from the rest.
        X, y = self.separate_targets(data, target_field)
        
        # Replace any missing values with a substitute.
        X = self.replace_missing_values(X, SubstitutionMethod.CLOSEST_NEIGHBOR_MEAN)
        
        # Normalize values by column.
        X, X_max, X_min = self.normalize_values(X)
        y, y_max, y_min = self.normalize_values(y)
        self.data_max = (X_max, y_max)
        self.data_min = (X_min, y_min)

        # Determine the target distribution for categorizing targets.
        self.target_dist = self.get_target_distribution(y)

        n = X.shape[0]
        d = X.shape[1]
        k = self.num_classes
        y = [self.categorize_target(t) for t in y]        
        X = [numpy.reshape(x, (d, 1)) for x in X]
        y = [numpy.reshape(t, (k, 1)) for t in y]
        self.num_features = d
        self.data = [(X[i], y[i]) for i in range(n)]
        
        # Separate the training data from the test data.
        test_size = int(len(self.data) * 0.3)
        self.train = self.data[0:-test_size]
        self.test = self.data[-test_size:]
        
    def get_description(self):
        return ''
    
    def read_csv(self, filepath, empty_value):
        data = []
        
        # Open file for reading.
        with open(filepath, 'rb') as input_file:
            # Use a CSV dictionary reader to read each desired field
            reader = csv.DictReader(input_file)
            for row in reader:
                entry = []
                for field in self.fields:
                    # If the field is empty, put a nan value in it.
                    if row[field] != '' and row[field] != empty_value:
                        entry.append(row[field])
                    else:
                        entry.append(numpy.nan)
                data.append(entry)
        return numpy.asarray(data)
    
    def split_categorical_fields(self, data, cat_fields):
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
        max_vals = numpy.amax(data, axis=0)
        min_vals = numpy.amin(data, axis=0)
        data = (data - min_vals) / (max_vals - min_vals)
        return data, max_vals, min_vals
        
    def separate_targets(self, data, field):
        target_column = self.fields.index(field)

        X = numpy.copy(data)
        y = numpy.copy(data[:,target_column])
        X = numpy.delete(X, [target_column], axis=1)

        return (X, y)
    
    def get_target_distribution(self, targets):
        target_dist = []
        partition_size = numpy.int(targets.shape[0]/(self.num_classes))
        partitions = self.partition(numpy.sort(targets), partition_size)
        for partition in partitions:
            target_dist.append(partition[0])
        return target_dist

    def partition(self, data, n):
        for i in range(0, len(data), n):
            yield data[i:i + n]

    def categorize_target(self, target):
        # Determine which class the value falls into, according to the 
        # target distribution.
        target_class = 0
        for j in range(self.num_classes):
            if target > self.target_dist[j]:
                target_class = j
        
        y = numpy.zeros(self.num_classes)
        y[target_class] = 1.0

        return y
            
class RedfinData(HousingData):
    def __init__(self, filepath, num_classes):
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
                target_field,
                num_classes,
                cat_fields=cat_fields)
        
    def get_description(self):
        return ('Housing data for the Grand Rapids, MI area.' + os.linesep +
                'Training data entries: {}'.format(len(self.train)) + os.linesep +
                'Test data entries: {}'.format(len(self.test)) + os.linesep +
                'Number of features: {}'.format(self.num_features) + os.linesep +
                'Target classes: {}'.format(self.target_dist))

class KingCountyData(HousingData):
    def __init__(self, filepath, num_classes):
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
        target_field = 'price'
        super(KingCountyData, self).__init__(
                filepath, 
                fields,
                target_field,
                num_classes)
        
    def get_description(self):
        return ('Housing data for the King County, WA area.' + os.linesep +
                'Training data entries: {}'.format(len(self.train)) + os.linesep +
                'Test data entries: {}'.format(len(self.test)) + os.linesep +
                'Number of features: {}'.format(self.num_features) + os.linesep +
                'Target classes: {}'.format(self.target_dist))

class NashvilleData(HousingData):
    def __init__(self, filepath, num_classes):
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
                target_field,
                num_classes,
                cat_fields=cat_fields)

    def get_description(self):
        """Gets a description of the housing data.
        """
        return ('Housing data for the Nashville, TN area.' + os.linesep +
                'Training data entries: {}'.format(len(self.train)) + os.linesep +
                'Test data entries: {}'.format(len(self.test)) + os.linesep +
                'Number of features: {}'.format(self.num_features) + os.linesep +
                'Target classes: {}'.format(self.target_dist))
        
class ARTData(HousingData):
    def __init__(self, filepath, num_classes):
        fields = [
                'MSSubClass',
                'MSZoning',
                'LotFrontage',
                'LotArea',
                'Street',
                'Alley',
                'LotShape',
                'LandContour',
                'Utilities',
                'LotConfig',
                'LandSlope',
                'Neighborhood',
                'Condition1',
                'Condition2',
                'BldgType',
                'HouseStyle',
                'OverallQual',
                'OverallCond',
                'YearBuilt',
                'YearRemodAdd',
                'RoofStyle',
                'RoofMatl',
                'Exterior1st',
                'Exterior2nd',
                'MasVnrType',
                'MasVnrArea',
                'ExterQual',
                'ExterCond',
                'Foundation',
                'BsmtQual',
                'BsmtCond',
                'BsmtExposure',
                'BsmtFinType1',
                'BsmtFinSF1',
                'BsmtFinType2',
                'BsmtFinSF2',
                'BsmtUnfSF',
                'TotalBsmtSF',
                'Heating',
                'HeatingQC',
                'CentralAir',
                'Electrical',
                '1stFlrSF',
                '2ndFlrSF',
                'LowQualFinSF',
                'GrLivArea',
                'BsmtFullBath',
                'BsmtHalfBath',
                'FullBath',
                'HalfBath',
                'BedroomAbvGr',
                'KitchenAbvGr',
                'KitchenQual',
                'TotRmsAbvGrd',
                'Functional',
                'Fireplaces',
                'FireplaceQu',
                'GarageType',
                'GarageYrBlt',
                'GarageFinish',
                'GarageCars',
                'GarageArea',
                'GarageQual',
                'GarageCond',
                'PavedDrive',
                'WoodDeckSF',
                'OpenPorchSF',
                'EnclosedPorch',
                '3SsnPorch',
                'ScreenPorch',
                'PoolArea',
                'PoolQC',
                'Fence',
                'MiscFeature',
                'MiscVal',
                'MoSold',
                'YrSold',
                'SaleType',
                'SaleCondition',
                'SalePrice'
                ]
        cat_fields = [
                'MSZoning',
                'Street',
                'Alley',
                'LotShape',
                'LandContour',
                'Utilities',
                'LotConfig',
                'LandSlope',
                'Neighborhood',
                'Condition1',
                'Condition2',
                'BldgType',
                'HouseStyle',
                'RoofStyle',
                'RoofMatl',
                'Exterior1st',
                'Exterior2nd',
                'MasVnrType',
                'ExterQual',
                'ExterCond',
                'Foundation',
                'BsmtQual',
                'BsmtCond',
                'BsmtExposure',
                'BsmtFinType1',
                'BsmtFinType2',
                'Heating',
                'HeatingQC',
                'CentralAir',
                'Electrical',
                'KitchenQual',
                'Functional',
                'FireplaceQu',
                'GarageType',
                'GarageFinish',
                'GarageQual',
                'GarageCond',
                'PavedDrive',
                'PoolQC',
                'Fence',
                'MiscFeature',
                'SaleType',
                'SaleCondition'
                ]
        target_field = 'SalePrice'
        empty_value = 'NA'
        super(ARTData, self).__init__(
                filepath,
                fields,
                target_field,
                num_classes,
                cat_fields=cat_fields,
                empty_value=empty_value)

    def get_description(self):
        return ('Housing data for Advanced Regression Techniques.' + os.linesep +
                'Training data entries: {}'.format(len(self.train)) + os.linesep +
                'Test data entries: {}'.format(len(self.test)) + os.linesep +
                'Number of features: {}'.format(self.num_features) + os.linesep +
                'Target classes: {}'.format(self.target_dist))