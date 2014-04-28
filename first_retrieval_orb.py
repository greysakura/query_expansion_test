__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np
import sys
from find_obj import filter_matches,explore_match


# def descriptor_distance_calc(vector_01, vector_02):
#     distance = np.dot((np.transpose(vector_01 - vector_02)),(vector_01 - vector_02))
#     return distance
if __name__ == "__main__":
    top_dir = 'C:/Cassandra/here/'
    # Number of clusters: 128 at present
    cluster_number = 256
    # Using SIFT here
    des_dimension = 32
    first_retrieval_num = 10
    # import target image
    target_image_dir = 'C:/Cassandra/new_orz.jpg'
    img = cv2.imread(target_image_dir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB(200)
    kpts_target, des_target = orb.detectAndCompute(img_gray, None)
    target_image_keypoint_num = len(kpts_target)

    img3 = cv2.drawKeypoints(img_gray,kpts_target,None,(0,0,255),4)
    cv2.imshow('img1', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print len(kpts_target)


    # Allocate each new descriptor to a certain cluster.

    kmeans_result_append = '/kmeans_result.txt'
    kmeans_result_dir = top_dir + kmeans_result_append
    kmeans_result_file = open(kmeans_result_dir, 'rb')

    line = kmeans_result_file.readline()
    cluster_number = int(line.split(',')[0])
    des_dimension = int(line.split(',')[1])
    print cluster_number
    print des_dimension

    centers = np.zeros((cluster_number, des_dimension), np.float32)

    for i in range(cluster_number):
        line = kmeans_result_file.readline()
        for j in range(des_dimension):
            centers[i,j] = np.float32(line.split(',')[j])
    kmeans_result_file.close()

    target_image_keypoint_labels = np.zeros((1,len(kpts_target)), np.int32)
    for i in range(len(kpts_target)):
        distance_calc = np.zeros((1,centers.shape[0]), np.float32)
        for j in range(centers.shape[0]):
            distance_calc[0,j] = np.dot((np.transpose(des_target[i] - centers[j])),(des_target[i] - centers[j]))
        ## argsort? we want to find the min distance, and that would be the nearest cluster center.
        target_image_keypoint_labels[0,i] = distance_calc.argsort(axis = 1)[0, 0]

    ## generate VW for target image.
    target_image_VW = np.zeros((1, cluster_number), np.int32)
    for i in range(target_image_keypoint_labels.shape[1]):
        target_image_VW[0, target_image_keypoint_labels[0,i]] += 1

    print target_image_VW

    # read the TF_IDF_matrix from file.
    # prepare for further VW matching under tf-idf structure
    TF_IDF_append = '/TF_IDF_matrix.txt'
    TF_IDF_dir = top_dir + TF_IDF_append
    TF_IDF_file = open(TF_IDF_dir, 'rb')
    line = TF_IDF_file.readline()
    # TF_IDF_matrix
    TF_IDF_matrix = np.zeros((int(line.split(',')[0]), int(line.split(',')[1])), np.float64)
    for i in range(TF_IDF_matrix.shape[0]):
        line = TF_IDF_file.readline()
        for j in range(TF_IDF_matrix.shape[1]):
            TF_IDF_matrix[i,j] = np.float64(line.split(',')[j])
    ## print TF_IDF_matrix
    TF_IDF_file.close()

    ## new image's descriptor file output
    target_image_des_append = '/target_image_python_des.csv'
    target_image_des_dir = top_dir + target_image_des_append
    that_file = open(target_image_des_dir, 'w')
    for i in range(des_target.shape[0]):
        for j in range(des_target.shape[1]):
            that_file.write(str(des_target[i, j]))
            if j < (des_target.shape[1]-1):
                that_file.write(',')
        that_file.write('\n')
    that_file.close()

    ## new image's kpts file output
    target_image_kpts_append = '/target_image_python_kpts.csv'
    target_image_kpts_dir = top_dir + target_image_kpts_append
    that_file = open(target_image_kpts_dir, 'w')
    for i in range(0, len(kpts_target) - 1):
        that_file.write(str(kpts_target[i].pt[0]))
        that_file.write(str(','))
        that_file.write(str(kpts_target[i].pt[1]))
        that_file.write(str(','))
        that_file.write(str(kpts_target[i].size))
        that_file.write(str(','))
        that_file.write(str(kpts_target[i].angle))
        that_file.write(str(','))
        that_file.write(str(kpts_target[i].response))
        that_file.write(str(','))
        that_file.write(str(kpts_target[i].octave))
        that_file.write(str(','))
        that_file.write(str(kpts_target[i].class_id))
        that_file.write('\n')
    that_file.close()

    ## new image's VW file output
    target_image_VW_append = '/target_image_python_VW.txt'
    target_image_VW_dir = top_dir + target_image_VW_append
    target_image_VW_file = open(target_image_VW_dir, 'w')
    target_image_VW_file.write(str(len(kpts_target)))
    target_image_VW_file.write(',')
    target_image_VW_file.write(str(cluster_number))
    target_image_VW_file.write('\n')
    for i in range(target_image_keypoint_labels.shape[1]):
        target_image_VW_file.write(str(target_image_keypoint_labels[0,i]))
        if i < (target_image_keypoint_labels.shape[1] - 1):
            target_image_VW_file.write(',')
    target_image_VW_file.write('\n')

    for i in range(target_image_VW.shape[1]):
        target_image_VW_file.write(str(target_image_VW[0,i]))
        if i < (target_image_VW.shape[1] - 1):
            target_image_VW_file.write(',')
    target_image_VW_file.write('\n')
    target_image_VW_file.close()

    ## here we do some extra 14/04/14
    ## we want to read the descriptors in the same VW. The top 3 VWs with the most descriptors.
    ## we set i as the number of VW we wanna count
    top_des_dump = []
    top_kpts_dump = []
    how_many_top = 3
    for i in range(how_many_top):
        des_tmp = np.zeros((target_image_VW[0,target_image_VW.argsort(axis = 1)[0,(-1-i)]], des_dimension), np.float32)
        top_kpts_tmp = []

        count_tmp = 0
        for j in range(target_image_keypoint_labels.shape[1]):
            if target_image_keypoint_labels[0,j] == target_image_VW.argsort(axis = 1)[0, (-1-i)]:
                ## for descriptors
                for k in range(des_dimension):
                    des_tmp[count_tmp, k] = des_target[j,k]
                ## for kpts
                kp_tmp = cv2.KeyPoint()
                kp_tmp.pt = (kpts_target[count_tmp].pt[0], kpts_target[count_tmp].pt[1])
                kp_tmp.size = float(kpts_target[count_tmp].size)
                kp_tmp.angle = float(kpts_target[count_tmp].angle)
                kp_tmp.response = float(kpts_target[count_tmp].response)
                kp_tmp.octave = int(float(kpts_target[count_tmp].octave))
                kp_tmp.class_id = int(float(kpts_target[count_tmp].class_id))
                top_kpts_tmp.append(kp_tmp)
                count_tmp += 1
        top_des_dump.append(des_tmp)
        top_kpts_dump.append(top_kpts_tmp)

    for i in range(how_many_top):
        print top_des_dump[i].shape
        print len(top_kpts_dump[i])

    ##########

    # now we calculate the "distance" between each database image and target image

    result_img_dir =[]
    result_img_kpts = []
    index_file = open('C:/Cassandra/here/image_index_python.txt','rb')
    image_count = 0
    for line in index_file:
        result_img_dir.append((line.split(','))[0])
        result_img_kpts.append(int(float(line.split(',')[1][:-2])))
    print result_img_dir
    print result_img_kpts
    index_file.close()
    image_count = len(result_img_dir)
    distance_between_image = np.zeros((1,image_count), np.float64)

    ## Use the right tf-idf Matrix!!!!!!!!  14/04/28
    tf_idf_store_dir = top_dir + '/TF_IDF_matrix.txt'
    tf_idf_store_file = open(tf_idf_store_dir, 'r')
    tf_idf_store_file.readline()
    for i in range(len(result_img_dir)):
        img_des_tmp = (result_img_dir[i].split('.'))[0] + '_VW.txt'
        the_file = open(img_des_tmp,'r')
        line = the_file.readline()
        line = the_file.readline()
        # read the third line
        line = the_file.readline()
        # get the VW of database image
        VW_tmp = np.zeros((1, cluster_number), np.int32)
        for j in range(cluster_number):
            VW_tmp[0, j] = int(float(line.split(',')[j]))
        the_file.close()
        # create a eye matrix with tf-idf values.
        TF_IDF_eye = np.zeros((cluster_number, cluster_number), np.float64)
        line_tf_idf_store = tf_idf_store_file.readline()
        for j in range(cluster_number):
            TF_IDF_eye[j, j] = np.float64(float(line_tf_idf_store.split(',')[j]))
        # calculate distance.
        distance_between_image[0, i] = np.dot((np.dot(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)), np.transpose(np.float64(target_image_VW - VW_tmp)))
    tf_idf_store_file.close()

    print distance_between_image
    print type(distance_between_image[0,0])
    distance_ranking = np.argsort(distance_between_image, axis=1)
    print distance_ranking

    for i in range(first_retrieval_num):
        print distance_between_image[0,distance_ranking[0,i]]
        img_tmp = cv2.imread(result_img_dir[distance_ranking[0][i]],0 )
        img_tmp = cv2.resize(img_tmp, (0,0), fx=0.5, fy=0.5)
        cv2.namedWindow(result_img_dir[distance_ranking[0][i]])
        cv2.imshow(result_img_dir[distance_ranking[0][i]], img_tmp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ## 14/04/28 here we've done first retrieval of image. But not good...
