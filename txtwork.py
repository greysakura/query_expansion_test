__author__ = 'LIMU_North'
import os
import csv
that_file=open('C:/Cassandra/testground.txt','w')

for i in range(0,10):
    that_file.write(str(i)+'\n')

that_file.close()