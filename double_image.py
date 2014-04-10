__author__ = 'LIMU_North'

import os
import cv
import cv2
import sys
import csv
import numpy as np

img1 = cv2.imread('C:/Cassandra/orz.jpg')
img2 = cv2.imread('C:/Cassandra/all_souls_000091.jpg')
img3 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
cv2.imshow('img1', img1)
print img1.shape
cv2.imshow('img2', img3)
print img2.shape
print type(img2)
img4 = np.zeros(((max(img3.shape[0], img1.shape[0])), (img3.shape[1] + img1.shape[1]),3), np.uint8)
# img4 = np.array((max(img1.shape[0], img2.shape[0])), (img1.shape[1] + img2.shape[1]), 3, dtype=np.int)

# left: 1
for i in range(0, img1.shape[0]):
    for j in range(0, img1.shape[1]):
        for k in range(0, 3):
            img4[i, j, k] = img1[i, j, k]
# right: 3
for i in range(0, img3.shape[0]):
    for j in range(0, img3.shape[1]):
        for k in range(0, 3):
            img4[i, (j+img1.shape[1]), k] = img3[i, j, k]

cv2.rectangle(img4, (0,0),(img1.shape[1],img1.shape[0]),(0,0,255),3)

cv2.rectangle(img4, (img1.shape[1], 0), (img1.shape[1] + img3.shape[1], img3.shape[0]),(255,0,0),3)

cv2.imshow('img4', img4)

cv2.waitKey(0)
cv2.destroyAllWindows()