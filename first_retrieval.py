__author__ = 'LIMU_North'

import csv
import cv2
import numpy as np


# def descriptor_distance_calc(vector_01, vector_02):
#     distance = np.dot((np.transpose(vector_01 - vector_02)),(vector_01 - vector_02))
#     return distance
if __name__ == "__main__":
    top_dir = 'C:/Cassandra/here/'

    # Number of clusters: 128 at present
    cluster_number = 128
    # Using SIFT here
    des_dimension = 128
    first_retrieval_num = 10

    # import target image

    target_image_dir = 'C:/Cassandra/orz.jpg'
    img = cv2.imread(target_image_dir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kpts_target, des = sift.detectAndCompute(img_gray, None)
    img = cv2.drawKeypoints(img_gray, kpts_target, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    target_image_keypoint_num = len(kpts_target)
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
            distance_calc[0,j] = np.dot((np.transpose(des[i] - centers[j])),(des[i] - centers[j]))
        target_image_keypoint_labels[0,i] = distance_calc.argmax(axis = 1)[0]

    # print target_image_keypoint_labels
    # print target_image_keypoint_labels.shape
    target_image_VW = np.zeros((1, cluster_number), np.int32)
    for i in range(target_image_keypoint_labels.shape[1]):
        target_image_VW[0, target_image_keypoint_labels[0,i]] += 1

    # print 'new_VW: ', target_image_VW
    # print 'new_VW.shape', target_image_VW.shape
    #
    # print target_image_VW
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
    # print TF_IDF_matrix
    TF_IDF_file.close()

    target_image_VW_append = '/target_image_VW_python.txt'
    target_image_VW_dir = top_dir + target_image_VW_append
    target_image_VW_file = open(target_image_VW_dir, 'wb')
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
        for j in range(cluster_number):
            TF_IDF_eye[j, j] = np.float64(VW_tmp[0, j])
        # calculate distance.
        distance_between_image[0, i] = np.dot((np.dot(np.float64(target_image_VW - VW_tmp), TF_IDF_eye)), np.transpose(np.float64(target_image_VW - VW_tmp)))

    print distance_between_image
    print type(distance_between_image[0,0])
    distance_ranking = np.argsort(distance_between_image, axis=1)
    print distance_ranking
    print distance_ranking[0][1]

    for i in range(first_retrieval_num):
        # img_tmp = cv2.imread(result_img_dir[distance_ranking[0][i]])
        # img_tmp = cv2.resize(img_tmp, (0,0), fx=0.5, fy=0.5)
        # cv2.namedWindow(result_img_dir[distance_ranking[0][i]])
        # cv2.imshow(result_img_dir[distance_ranking[0][i]], img_tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        str_kpts = '_kpts.csv'
        str_des = '_des.csv'
        str_VW = '_VW.txt'
        image_dir_kpts = (result_img_dir[distance_ranking[0][i]].split('.'))[0] + str_kpts
        image_dir_des = (result_img_dir[distance_ranking[0][i]].split('.'))[0] + str_des
        image_dir_VW = (result_img_dir[distance_ranking[0][i]].split('.'))[0] + str_VW
        tmp_VW_file = open(image_dir_VW, 'rb')
        line_VW = tmp_VW_file.readline()
        des_num_tmp = int(float(line_VW.split(',')[0]))
        des_dimension_tmp = int(float(line_VW.split(',')[1]))
        # tmp for descriptors
        des_tmp_file = open(image_dir_des,'rb')
        des_mat_tmp = np.zeros((des_num_tmp, des_dimension_tmp), np.int32)
        reader_des = csv.reader(des_tmp_file, delimiter=',', quoting = csv.QUOTE_NONE)
        row_count = 0
        for row in reader_des:
            for j in range(len(row)):
                des_mat_tmp[row_count, j] = int(float(row[j]))
            row_count += 1
        des_tmp_file.close()
        # tmp for kpts
        kpts_tmp_file = open(image_dir_kpts,'rb')
        kpts_mat_tmp = np.zeros((des_num_tmp, 7))
        kpts_tmp = []
        reader_kpts = csv.reader(kpts_tmp_file, delimiter=',', quoting = csv.QUOTE_NONE)
        for row in reader_kpts:
            kp_tmp = cv2.KeyPoint()
            kp_tmp.pt = (float(row[0]), float(row[1]))
            kp_tmp.size = float(row[2])
            kp_tmp.angle = float(row[3])
            kp_tmp.response = float(row[4])
            kp_tmp.octave = float(row[5])
            kp_tmp.class_id = int(float(row[6]))
            # kpts_tmp.append(kp_tmp)
        kpts_tmp_file.close()
        print kpts_tmp

        tmp_VW_file.close()


    # spatial verification stage




















