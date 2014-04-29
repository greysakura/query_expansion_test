__author__ = 'LIMU_North'

import os
import cv
import cv2
import sys
import csv
import numpy as np

top_dir = 'C:/Cassandra/here/'
query_goto_dir = 'C:/Cassandra/query_object/'
ground_truth_dir = top_dir + 'ground_truth_file/'
#
# Const_Image_Format = [".txt"]
# class FileFilt:
#     fileList = [""]
#     counter = 0
#     def __init__(self):
#         pass
#     def FindFile(self,dirr,filtrate = 1):
#         global Const_Image_Format
#         for s in os.listdir(dirr):
#             newDir = os.path.join(dirr,s)
#             if os.path.isfile(newDir):
#                 if filtrate:
#                         if newDir and(os.path.splitext(newDir)[1] in Const_Image_Format):
#                             self.fileList.append(newDir)
#                             self.counter += 1
#                 else:
#                     self.fileList.append(newDir)
#                     self.counter += 1


if __name__ == "__main__":

    # img1_dir = top_dir + 'all_souls_000013.jpg'
    # img1 = cv2.imread(img1_dir)
    # # cv2.rectangle(img1, (186, 163), (589, 859),(255,0,0),3)
    # cv2.imshow('img1', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # img2 = img1[34.1:955.7, 136.5:648.5]
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('C:/Cassandra/all_souls_01.jpg', img2)

    target_name = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', 'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']

    print target_name
    print len(target_name)
    ## check if the dir exsit. If not, make one.
    if not os.path.exists(query_goto_dir):
                os.makedirs(query_goto_dir)
    ##
    for i in range(len(target_name)):
        query_append = '_query.txt'
        for j in range(5):
            print
            query_file_tmp = open(ground_truth_dir + target_name[i] + '_' + str(j+1) + '_query.txt', 'r')
            line = query_file_tmp.readline()
            query_name_tmp = line.split(' ')[0][5:]
            print line[:-1]
            print 'query_name_tmp: ', query_name_tmp
            query_file_tmp.close()
            ## read the image file from dataset
            original_img_tmp = cv2.imread(top_dir + query_name_tmp + '.jpg')
            ## get the query region information
            c0 = float(line.split(' ')[2])
            c1 = float(line.split(' ')[4])
            r0 = float(line.split(' ')[1])
            r1 = float(line.split(' ')[3])

            query_object_img_tmp = original_img_tmp[c0:c1, r0:r1]
            query_object_dir_tmp = query_goto_dir + target_name[i] + '_' + str(j+1) + '_query.jpg'
            cv2.imshow(query_object_dir_tmp, query_object_img_tmp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            ## write down this query object image.

            cv2.imwrite(query_object_dir_tmp, query_object_img_tmp)




