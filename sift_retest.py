__author__ = 'LIMU_North'

import numpy as np
import cv
import os
import cv2
import find_obj
from matplotlib import pyplot as plt
from find_obj import filter_matches, explore_match

MIN_MATCH_COUNT = 10

# queryImage
img1 = cv2.imread('C:/Cassandra/orz_grey.jpg',0)
# trainImage
img2 = cv2.imread('C:/Cassandra/hereafter/grey_image12.jpg',0)
orb = cv2.ORB()



sift = cv2.SIFT()

kp1, des1 = orb.detectAndCompute(img1, None)

kp2, des2 = orb.detectAndCompute(img2, None)

des1 = np.uint8(des1)
des2 = np.uint8(des2)
print type(des2[0,0])
print des1.shape
print des1[:,0]

for i in range(len(kp1)):
    # kp1[i].octave = (kp1[i].octave & 0xFF)
    kp1[i].octave = (kp1[i].octave % 256)
    if kp1[i].octave > 8:
        kp1[i].octave = (kp1[i].octave - 256)
    print 'kp1[', i , '].octave: ', kp1[i].octave

print 'length of kp1: ', len(kp1)
print 'length of kp2: ', len(kp2)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)#, crossCheck=True)
matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)

img3 = cv2.drawKeypoints(img1,kp1,color=(0,0,255), flags=0)
cv2.imshow('img1', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

p1, p2, kp_pairs = filter_matches(kp1, kp2, matches, 0.95)
if len(kp_pairs):
    explore_match('find_obj', img1,img2,kp_pairs)#cv2 shows image

    cv2.waitKey()
    cv2.destroyAllWindows()

real_tmp = np.zeros((0,0), np.int32)

print real_tmp.shape

