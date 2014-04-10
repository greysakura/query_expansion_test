__author__ = 'LIMU_North'
import os
import csv

csvfile = file('C:/Cassandra/testcsv.csv', 'rb')
reader = csv.reader(csvfile)
# line = reader.readline()
for line in reader:
    print line

csvfile.close()

# for item in reader:
#     print item