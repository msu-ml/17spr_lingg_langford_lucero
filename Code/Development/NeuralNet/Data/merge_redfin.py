# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:08:06 2017

@author: Michael Austin Langford
"""
import csv
import getopt
import os
import re
import sys

def main(argv):
    filepaths = [
            'redfin_mi_allendale_000_up.csv',
            'redfin_mi_byroncenter_000_up.csv',
            'redfin_mi_caledonia_000_up.csv',
            'redfin_mi_cannonburg_000_up.csv',
            'redfin_mi_comstockpark_000_up.csv',
            'redfin_mi_coopersville_000_up.csv',
            'redfin_mi_cutlerville_000_up.csv',
            'redfin_mi_eastgrandrapids_000_up.csv',
            'redfin_mi_foresthills_000_300.csv',
            'redfin_mi_foresthills_300_up.csv',
            'redfin_mi_georgetown_000_up.csv',
            'redfin_mi_grandrapids_000_050.csv',
            'redfin_mi_grandrapids_050_075.csv',
            'redfin_mi_grandrapids_075_100.csv',
            'redfin_mi_grandrapids_100_125.csv',
            'redfin_mi_grandrapids_125_150.csv',
            'redfin_mi_grandrapids_150_175.csv',
            'redfin_mi_grandrapids_175_200.csv',
            'redfin_mi_grandrapids_200_250.csv',
            'redfin_mi_grandrapids_250_350.csv',
            'redfin_mi_grandrapids_350_up.csv',
            'redfin_mi_grandville_000_up.csv',
            'redfin_mi_kentwood_000_125.csv',
            'redfin_mi_kentwood_125_150.csv',
            'redfin_mi_kentwood_150_175.csv',
            'redfin_mi_kentwood_175_275.csv',
            'redfin_mi_kentwood_275_up.csv',
            'redfin_mi_lowell_000_up.csv',
            'redfin_mi_negrandrapids_000_150.csv',
            'redfin_mi_negrandrapids_150_250.csv',
            'redfin_mi_negrandrapids_250_up.csv',
            'redfin_mi_nwgrandrapids_000_up.csv',
            'redfin_mi_rockford_000_up.csv',
            'redfin_mi_walker_000_200.csv',
            'redfin_mi_walker_200_up.csv',
            'redfin_mi_wyoming_000_075.csv',
            'redfin_mi_wyoming_075_100.csv',
            'redfin_mi_wyoming_100_125.csv',
            'redfin_mi_wyoming_125_150.csv',
            'redfin_mi_wyoming_150_250.csv',
            'redfin_mi_wyoming_250_up.csv',
            ]

    fields = ['PROPERTY TYPE',
               'ADDRESS',
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
    
    data = []
    data.append(fields)
    for filepath in filepaths:
        with open(filepath, 'rb') as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                entry = []
                for field in fields:
                    entry.append(row[field])
                if (row['PRICE'] != ''):
                    data.append(entry)
   
    print('Entries: {}'.format(len(data)))
    
    encoded_fields = [('PROPERTY TYPE', 0),
                      ('ADDRESS', 1),
                      ('CITY', 2),
                      ('STATE', 3)
                      ]
    
    with open('redfin.csv', 'wb') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)
    
    encodings = {}
    for (field, idx) in encoded_fields:
        encodings[field] = {}
        data[1:] = encode(data[1:], encodings[field], idx)
    
    with open('redfin_encoded.csv', 'wb') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(data)
        
    with open('redfin_encodings.txt', 'wt') as output_file:
        for (field, idx) in encoded_fields:
            output_file.write(field + os.linesep)
            keys = sorted(encodings[field].keys(), key=str.lower)
            for key in keys:
                output_file.write('{} : {}'.format(encodings[field][key], key) + os.linesep)
            output_file.write(os.linesep)

def encode(data, encodings, idx):
    regex = re.compile(r"""\s*(\d|\-|\&|\s)*\s*(?P<value>(\w\s*)+)\s*(#\d+)?\s*""")
    for entry in data:
        match = regex.match(entry[idx])
        if match != None:
            value = match.group('value').title()
            if not value in encodings.keys():
                encodings[value] = len(encodings) + 1
            entry[idx] = encodings[value]
        else:
            entry[idx] = ''
    return data

if __name__ == '__main__':
    main(sys.argv[1:])