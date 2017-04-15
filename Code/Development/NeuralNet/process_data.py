# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 09:55:15 2017

@author: Michael Austin Langford
"""

import getopt
import numpy as np
import os
import re
import sys
from HousingData import HousingData
from HousingData import SubstitutionMethod

seed = 69
np.random.seed(seed)

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
                'f:o:1:2:x:ns:',
                ['fields=', 'output=', 'maxy=', 'miny=', 'missing=', 'normalize', 'sub='])
    except getopt.GetoptError:
       sys.exit(2)

    filepath = ''
    output_filepath = ''
    fields = []
    cat_fields = []
    empty_value = ''
    normalize = False
    subMethod = SubstitutionMethod.NONE
    y_min = None
    y_max = None
    target_bounds = None

    for opt, arg in opts:
        if opt in ('-o', '--output'):
            output_filepath = arg
        elif opt in ('-f', '--fields'):
            fields, cat_fields = read_fields(arg)
        elif opt in ('-x', '--missing'):
            empty_value = arg
        elif opt in ('-n', '--normalize'):
            normalize = True
        elif opt in ('-s', '--sub'):
            method = arg.lower()
            if method == 'mean':
                subMethod = SubstitutionMethod.MEAN
            elif method == 'closest_value':
                subMethod = SubstitutionMethod.CLOSEST_VALUE
            elif method == 'closest_mean':
                subMethod = SubstitutionMethod.CLOSEST_MEAN
        elif opt in ('-1', '--miny'):
            y_min = np.float32(arg)
        elif opt in ('-2', '--maxy'):
            y_max = np.float32(arg)

    if len(args) > 0:
        filepath = args[0]
        if output_filepath == '':
            output_filepath = os.path.splitext(filepath)[0] + '_processed.csv'
    
        if (not y_min is None and not y_max is None):
            target_bounds = (y_min, y_max)

        print('Processing data...')
        data = HousingData(
                filepath,
                preprocessed=False,
                fields=fields,
                cat_fields=cat_fields,
                empty_value=empty_value,
                subMethod=subMethod,
                normalize=normalize,
                target_bounds=target_bounds)
        
        print('Writing processed data...')
        data.write_csv(output_filepath)

if __name__ == '__main__':
    main(sys.argv[1:])
