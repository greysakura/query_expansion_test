__author__ = 'LIMU_North'

import cv2
import numpy as np

img = cv2.imread('C:/Cassandra/000001.jpg')

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img,img, mask= mask)

cv2.imshow('img',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.waitKey(0)