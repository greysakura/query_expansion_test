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



print type(img1)

# img3 = cv2.resize(img2,(0,0), fx=0.5, fy=0.5)

# Initiate SIFT detector
orb = cv2.ORB()
kp_tmp = cv2.KeyPoint()
print type(kp_tmp.size)
kp_tmp.pt = (1,1)
print kp_tmp.pt

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

print type(kp1[0].pt[0])
print type(kp1[0].octave)
print 'octave: ', kp1[1].octave

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING)#, crossCheck=True)

matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)
explore_match('find_obj', img1,img2,kp_pairs)#cv2 shows image

cv2.waitKey()
cv2.destroyAllWindows()


