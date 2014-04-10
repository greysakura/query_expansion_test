__author__ = 'LIMU_North'
import cv2
import numpy as np
finish = False
draw_ok = False
ix,iy = -1,-1
longer_x = -1
longer_y = -1
origin_x = -1
origin_y = -1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy, finish, longer_x, longer_y, origin_x, origin_y, draw_ok
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        if(x != ix) | (y != iy):
            draw_ok = True
            cv2.rectangle(img,(ix,iy),(x,y),(0,0,255),3)
            longer_x = abs(x-ix)
            longer_y = abs(y-iy)
            origin_x = min(x, ix)
            origin_y = min(y, iy)
            finish = True






# Create a black image, a window and bind the function to window
img = cv2.imread('C:/Cassandra/all_souls_000091.jpg')

img_copy= cv2.imread('C:/Cassandra/all_souls_000091.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if (cv2.waitKey(20) & 0xFF == 27) | (finish == True):
        break

cv2.destroyAllWindows()
if draw_ok == True:
    # print longer_x, longer_y, origin_x, origin_y
    img_tmp = np.zeros((longer_y, longer_x, 3), np.uint8)
    for i in range(0, longer_y):
        for j in range(0, longer_x):
            for k in range(0, 3):
                img_tmp[i, j, k] = img_copy[(i+origin_y), (j+origin_x), k]
    ratio_x = 0.5
    ratio_y = 0.5
    cv2.imshow('new_image', img_tmp)
    # cv2.rectangle(img,(ix,iy),(x,y),(0,0,255),3)
    cv2.imshow('origin image', img)
    img = cv2.resize(img, (0,0), fx=ratio_x, fy=ratio_y)
    img_final = np.zeros(((max(img_tmp.shape[0], img.shape[0])), (img_tmp.shape[1] + img.shape[1]),3), np.uint8)

    # left: 1 tmp
    for i in range(0, img_tmp.shape[0]):
        for j in range(0, img_tmp.shape[1]):
            for k in range(0, 3):
                img_final[i, j, k] = img_tmp[i, j, k]
    # right: 3 img
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            for k in range(0, 3):
                img_final[i, (j+img_tmp.shape[1]), k] = img[i, j, k]

    cv2.rectangle(img_final, (0,0), (img_tmp.shape[1], img_tmp.shape[0]),(0,0,255),3)
    cv2.line(img_final, (0,0),(img_tmp.shape[1] + int(origin_x*ratio_x), int(origin_y*ratio_y)),(0,0,255), 2, 4, 0 )
    cv2.line(img_final, (0,img_tmp.shape[0]),(img_tmp.shape[1] + int(origin_x*ratio_x), int(origin_y*ratio_y)+int(longer_y*ratio_y)),(0,0,255), 2, 4, 0 )
    cv2.line(img_final, (img_tmp.shape[1],img_tmp.shape[0]),(img_tmp.shape[1] + int(origin_x*ratio_x) + int(longer_x*ratio_x),
                                                             int(origin_y*ratio_y)+int(longer_y*ratio_y)),(0,0,255), 2, 4, 0 )
    cv2.line(img_final, (img_tmp.shape[1],0),(img_tmp.shape[1] + int(origin_x*ratio_x) + int(longer_x*ratio_x), int(origin_y*ratio_y)),(0,0,255), 2, 4, 0 )

    cv2.imshow('two', img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()