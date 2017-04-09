# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:08:06 2017

@author: Michael Austin Langford
"""
import csv
import os
import numpy
import sys
import time
from geopy import geocoders

def main(argv):
    fields = ['Unnamed: 0']
    
    geocodes = []
    with open('Nashville_geocoded.csv', 'rb') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            geocodes.append(row)
    
    data = []
    with open('Nashville_housing_data_2013_2016.csv', 'rb') as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            entry = row[:]
            try:
                num = int(entry[0])
                for field in geocodes[num+1][1:]:
                    entry.append(field)
            except:
                for field in geocodes[0][1:]:
                    entry.append(field)
            data.append(entry)
            
    with open('Nashville_housing_data_2013_2016_geocoded.csv', 'wb') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)

if __name__ == '__main__':
    main(sys.argv[1:])
