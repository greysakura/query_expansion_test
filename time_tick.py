__author__ = 'LIMU_North'

import cv2
import numpy as np

img1 = cv2.imread('C:/Cassandra/000001.jpg')

e1 = cv2.getTickCount()
for i in xrange(5,49,2):
    img1 = cv2.medianBlur(img1,i)
e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print t

cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

print cv2.useOptimized()