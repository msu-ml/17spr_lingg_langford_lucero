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

def partitions(data, n):
    for i in xrange(0, len(data), n):
        yield data[i:i + n]

def main(argv):
    filepath = 'Nashville_housing_data_2013_2016.csv'
    
    fields = ['Unnamed: 0',
              'Property Address',
              'Property City'
             ]
    
    data = []
    data.append(fields)
    with open(filepath, 'rb') as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            entry = []
            for field in fields:
                entry.append(row[field])
            entry.append('TN')
            entry.append('')
            data.append(entry)

"""
    curr = 0
    for partition in partitions(data[1:], 1000):
        with open('Nashville_housing_data_{}.csv'.format(curr), 'wb') as output_file:
            writer = csv.writer(output_file)
            writer.writerows(partition)
        curr = curr + 1
"""

    with open('Nashville_housing_data_2013_2016_geocoded.txt', 'wt') as output_file:
        geocoder = geocoders.GoogleV3()
        for entry in data[1:]:
            num = entry[0]
            addr = entry[1]
            city = entry[2]
            state = 'TN'
            if (int(num) > 1582):
                full_addr = '{}, {}, {}'.format(addr, city, state)
                geocoded = False
                while not geocoded:
                    try:
                        place, (lat, lng) = geocoder.geocode(full_addr)
                        geocoded = True
                    except:
                        time.sleep(2)
                print('{}: {}, {}, {} -> {} ({}, {})'.format(num, addr, city, state, place, lat, lng))
                output_file.write('{}\t{}\t{}\t{}'.format(num, place, lat, lng) + os.linesep)

if __name__ == '__main__':
    main(sys.argv[1:])
