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
from housing_data import HousingData
from housing_data import SubstitutionMethod

seed = 69
numpy.random.seed(seed)

def read_fields(field_filepath):
        fields = []
        cat_fields = []
        
        with open(field_filepath) as input_file:
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

def main(argv):
    try:                                
        opts, args = getopt.getopt(
                argv,
                'f:o:x:ns:',
                ['fields', 'output', 'missing', 'normalize', 'sub'])
    except getopt.GetoptError:
        sys.exit(2)

    filepath = ''
    output_filepath = ''
    fields = []
    cat_fields = []
    empty_value = ''
    normalize = False
    subMethod = SubstitutionMethod.NONE

    for opt, arg in opts:
        if opt in ('-o', '--output'):
            output_filepath = arg
        elif opt in ('-f', '--fields'):
            fields, cat_fields = read_fields(arg)
        elif opt in ('-x', '--missing'):
            empty_value = arg
        elif opt in ('-n', '--normalize'):
            normalize = True
        elif opt in ('-s', '-sub'):
            method = arg.lower()
            if method == 'mean':
                subMethod = SubstitutionMethod.MEAN
            elif method == 'closest_value':
                subMethod = SubstitutionMethod.CLOSEST_VALUE
            elif method == 'closest_mean':
                subMethod = SubstitutionMethod.CLOSEST_MEAN

    if len(args) > 0:
        filepath = args[0]
        if output_filepath == '':
            output_filepath = os.path.splitext(filepath)[0] + '_processed.csv'
    
        print('Processing data...')
        data = HousingData(
                filepath,
                fields=fields,
                cat_fields=cat_fields,
                empty_value=empty_value,
                subMethod=subMethod,
                normalize=normalize,
                preprocessed=False)
        
        print('Writing processed data...')
        data.write_csv(output_filepath)

if __name__ == '__main__':
    main(sys.argv[1:])
