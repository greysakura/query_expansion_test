"""
__author__ = 'LIMU_North'
"""


import math
import os
import cv
import cv2
import sys
import csv

# print "Hello, I'm Olaf. I like warm hugs."



# # queryImage
# img1 = cv2.imread('C:/Cassandra/orz_grey.jpg')
# print type(img1)
# img3 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
# print type(img3)
# # trainImage
# img2 = cv2.imread('C:/Cassandra/hereafter/grey_image12.jpg',0)
#
# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
# cv2.imshow('img3', img3)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread('C:/Cassandra/orz_grey.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg', img)

image_dir = 'C:/Cassandra/orz_grey.jpg'

print type(image_dir)

print image_dir[:-4]
str_kpts = '_kpts.csv'
str_des = '_des.csv'
image_dir_kpts = image_dir[:-4] + str_kpts
image_dir_des = image_dir[:-4] + str_des
print image_dir_kpts


that_file = open(image_dir_des, 'w')
# csv_file = file('csv_test.csv', 'wb')
# writer = open(csv_file)

for i in range(0, des.shape[0]-1):
    for j in range(0, des.shape[1]-1):
        that_file.write(str(des[i, j]))
        if j < (des.shape[1]-1):
            that_file.write(',')
    that_file.write('\n')

that_file.close()

# cv2.imshow('image', img)
# cv2.waitKey(0)

print "kp.x: ", kp[0].pt[0]
print "kp.y: ", kp[0].pt[1]
print "kp.size:", kp[0].size
print "kp.angle:", kp[0].angle
print "kp.response:", kp[0].response
print "kp.octave:", kp[0].octave
print "kp.class_id", kp[0].class_id

print len(kp)



that_file = open(image_dir_kpts, 'w')
# csv_file = file('csv_test.csv', 'wb')
# writer = open(csv_file)

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





