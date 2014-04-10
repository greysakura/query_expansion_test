__author__ = 'LIMU_North'
import os
import numpy as np

result=[]
that_file = open('C:/Cassandra/image_index.txt', 'r')
line = that_file.readline()

for line in that_file:
    result.append(map(str,line.split(' ')))
print(result)

a = np.mat('1 2 3; 4 5 3')

print a

print a[0,1]

b = np.zeros((5,4), dtype=np.int)

b[3,3] = int(1223.23)

print b