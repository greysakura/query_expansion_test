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

surf = cv2.SURF(5000)
surf02 = cv2.SURF(10000)
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf02.detectAndCompute(img2,None)
print des1[0].shape
print len(des1)

for i in range(len(kp1)):
    print kp1[i].octave
img3 = cv2.drawKeypoints(img1,kp1,None,(255,0,0),4)
img4 = cv2.drawKeypoints(img2,kp2,None,(0,0,255),4)
print len(kp2)
cv2.imshow('img2', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()