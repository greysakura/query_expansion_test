__author__ = 'LIMU_North'

import os
import cv
import cv2
import sys
import csv
import numpy as np

Const_Image_Format = [".jpg", ".bmp", ".png"]
class FileFilt:
    fileList = [""]
    counter = 0
    def __init__(self):
        pass
    def FindFile(self,dirr,filtrate = 1):
        global Const_Image_Format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr,s)
            if os.path.isfile(newDir):
                if filtrate:
                        if newDir and(os.path.splitext(newDir)[1] in Const_Image_Format):
                            self.fileList.append(newDir)
                            self.counter += 1
                else:
                    self.fileList.append(newDir)
                    self.counter += 1


def search_dir_and_create_csv(image_dir, top_dir):
    keypoint_num = 0
    img = cv2.imread(image_dir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (0,0), fx=0.5, fy=0.5)
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(img_gray, None)
    img = cv2.drawKeypoints(img_gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    keypoint_num = len(kp)
    print image_dir
    print keypoint_num
    #create dirs for kpts and descriptors
    str_kpts = '_kpts.csv'
    str_des = '_des.csv'
    image_dir_kpts = (image_dir.split('.'))[0] + str_kpts
    image_dir_des = (image_dir.split('.'))[0] + str_des

    that_file = open(image_dir_des, 'w')
    for i in range(des.shape[0]):
        for j in range(des.shape[1]):
            that_file.write(str(des[i, j]))
            if j < (des.shape[1]-1):
                that_file.write(',')
        that_file.write('\n')
    that_file.close()

    that_file = open(image_dir_kpts, 'w')
    for i in range(0, len(kp) - 1):
        that_file.write(str(kp[i].pt[0]))
        that_file.write(str(','))
        that_file.write(str(kp[i].pt[1]))
        that_file.write(str(','))
        that_file.write(str(kp[i].size))
        that_file.write(str(','))
        that_file.write(str(kp[i].angle))
        that_file.write(str(','))
        that_file.write(str(kp[i].response))
        that_file.write(str(','))
        that_file.write(str(kp[i].octave))
        that_file.write(str(','))
        that_file.write(str(kp[i].class_id))
        that_file.write('\n')
    that_file.close()
    return keypoint_num

if __name__ == "__main__":
    top_dir = 'C:/Cassandra/here/'

    str_image_index_python_append = '/image_index_python.txt'
    str_image_index = top_dir + str_image_index_python_append
    file_image_index = open(str_image_index, 'w')
    image_search_dir = FileFilt()
    image_search_dir.FindFile(dirr=top_dir)
    print(image_search_dir.counter)
    for image_dir in image_search_dir.fileList:
        # print "image_dir: ", image_dir_input
        # print type(image_dir_input)
        if len(image_dir) != 0:
            keypoint_num = search_dir_and_create_csv(image_dir, top_dir)
            file_image_index.write(image_dir)
            file_image_index.write(',')
            file_image_index.write(str(keypoint_num))
            file_image_index.write('\n')

    file_image_index.close()