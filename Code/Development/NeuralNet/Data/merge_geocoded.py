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
    filepaths = []
    for i in range(57):
        filepaths.append('GeocodeResults ({}).csv'.format(i))
        
    data = []
    for filepath in filepaths:
        with open(filepath, 'rb') as input_file:
            reader = csv.reader(input_file)
            for row in reader:
                num = int(row[0])
                try:
                    regex = re.compile(r"""\s*(?P<addr>(\w\s*)+),\s*(?P<city>(\w\s*)+),\s*(?P<state>(\w\s*)+),\s*(?P<zipcode>(\w\s*)+)\s*""")
                    match = regex.match(row[4])
                    addr = match.group('addr').title()
                    city = match.group('city').title()
                    state = match.group('state').upper()
                    zipcode = int(match.group('zipcode'))
                except:
                    addr = ''
                    city = ''
                    state = ''
                    zipcode = ''
                try:
                    regex = re.compile(r"""\s*(?P<lat>(\-|\d|\.)+),\s*(?P<lng>(\-|\d|\.)+)\s*""")
                    match = regex.match(row[5])
                    lat = float(match.group('lat'))
                    lng = float(match.group('lng'))
                except:
                    lat = ''
                    lng = ''
                entry = [num, addr, city, state, zipcode, lat, lng]
                data.append(entry)
    
    data = sorted(data)
        
    with open('Nashville_geocoded.csv', 'wb') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['geocoded_address', 'geocoded_city', 'geocoded_state', 'geocoded_zipcode', 'geocoded_latitude', 'geocoded_longitude'])
        writer.writerows(data)

if __name__ == '__main__':
    main(sys.argv[1:])