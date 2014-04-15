__author__ = 'LIMU_North'
import numpy as np
import cv2
import find_obj
from matplotlib import pyplot as plt
from find_obj import filter_matches, explore_match

MIN_MATCH_COUNT = 10

# queryImage
img1 = cv2.imread('C:/Cassandra/orz_grey.jpg')
img3 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# trainImage
img2 = cv2.imread('C:/Cassandra/here/all_souls_000013.jpg')
img4 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


surf = cv2.SURF(2500)
surf02 = cv2.SURF(5000)
sift = cv2.SIFT()
kp = surf.detect(img1,None)
kp1, des1 = sift.compute(img1,kp)
# kp1, des1 = surf.detectAndCompute(img1,None)
kp2 = surf02.detect(img4, None)

kp2, des2 = sift.compute(img4,kp2)

print type(des1)
print des1
print len(des1[0])
print len(des1)

# for i in range(len(kp1)):
#     print kp1[i].octave
img5 = cv2.drawKeypoints(img1,kp1,None,(255,0,0),4)
img6 = cv2.drawKeypoints(img2,kp2,None,(0,0,255),4)
print len(kp1)
cv2.imshow('img2 kpts detected', img6)
cv2.waitKey(0)
cv2.destroyAllWindows()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.99*n.distance:
        good.append(m)
print len(good)

# a = np.arange(6).reshape((3, 2))
# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
# print src_pts
# np.reshape(src_pts, (-1,1,2))
# print src_pts
# print type(src_pts)
# print type(a)
#
# print a

MIN_MATCH_COUNT = 5
if len(good)>MIN_MATCH_COUNT:
    print 'length of good: ', len(good)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    np.reshape(src_pts, (-1,1,2))
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
    np.reshape(dst_pts, (-1,1,2))
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    print M
    matchesMask = mask.ravel().tolist()
    h,w = img3.shape
    print h, w
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ])
    np.reshape(pts, (-1,1,2))

    # the step below is the key. Turn it into 3-dimensionals
    pts = np.array([pts])
    #

    dst = cv2.perspectiveTransform(pts,M)
    print 'dst.shape: ', dst.shape
    # print np.int32(dst)
    cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),5)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None


# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()