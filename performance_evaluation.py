__author__ = 'LIMU_North'

import os
import cv
import cv2
import sys
import csv
import numpy as np

top_dir = 'C:/Cassandra/here/'
ground_truth_dir = top_dir + 'ground_truth_file/'

if __name__ == "__main__":
    our_file_name = 'all_souls_000005'
    test_name = 'all_souls_1'
    good_stack = 0
    ok_stack = 0
    junk_stack = 0
    for i in range(3):
        if i == 0:
            ground_truth_file_dir = ground_truth_dir + test_name + '_good.txt'
            print ground_truth_file_dir, ': Good'
        elif i == 1:
            ground_truth_file_dir = ground_truth_dir + test_name + '_ok.txt'
            print ground_truth_file_dir, ': OK'
        else:
            ground_truth_file_dir = ground_truth_dir + test_name + '_junk.txt'
            print ground_truth_file_dir, ': JUNK'
        # print ground_truth_file_dir
        ground_truth_file = open(ground_truth_file_dir, 'r')

        for line in ground_truth_file:
            print line[:-1]
            print line.find(our_file_name)
            if line.find(our_file_name) == 0:
                if i == 0:
                    good_stack += 1
                elif i == 1:
                    ok_stack += 1
                else:
                    junk_stack += 1



        ground_truth_file.close()
        print 'good_stack: ', good_stack
        print 'ok_stack: ', ok_stack
        print 'junk_stack: ', junk_stack

