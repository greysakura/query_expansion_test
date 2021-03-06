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
    cluster_number = 128
    # Using SIFT here
    des_dimension = 128
    first_retrieval_num = 10
    ## number of VW for grouping
    how_many_top = 3

    # import target image
    target_image_dir = 'C:/Cassandra/orz.jpg'
    img = cv2.imread(target_image_dir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.SURF(2000)
    sift = cv2.SIFT()
    kpts_target = surf.detect(img_gray, None)
    kpts_target, des_target = sift.compute(img_gray, kpts_target)
    # kpts_target, des_target = sift.detectAndCompute(img_gray, None)
    # img = cv2.drawKeypoints(img_gray, kpts_target, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    target_image_keypoint_num = len(kpts_target)

    img3 = cv2.drawKeypoints(img_gray,kpts_target,None,(0,0,255),4)
    cv2.imshow('img1', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print len(kpts_target)

    # for i in range(len(kpts_target)):
    #     kpts_target[i].octave = (kpts_target[i].octave % 256)
    #     if kpts_target[i].octave > 8:
    #         kpts_target[i].octave = (kpts_target[i].octave - 256)
    #     print kpts_target[i].octave


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
        target_image_keypoint_labels[0,i] = distance_calc.argsort(axis = 1)[0,0]

    # print target_image_keypoint_labels
    # print target_image_keypoint_labels.shape
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
    target_image_top_des = []
    target_image_top_kpts = []

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
        target_image_top_des.append(des_tmp)
        target_image_top_kpts.append(top_kpts_tmp)

    for i in range(how_many_top):
        print target_image_top_des[i].shape
        print len(target_image_top_kpts[i])

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


    ## i, image numbers in first retrieval
    for i in range(first_retrieval_num):
        img_tmp = cv2.imread(result_img_dir[distance_ranking[0][i]],0 )
        img_tmp = cv2.resize(img_tmp, (0,0), fx=0.5, fy=0.5)
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

        ## for VW
        tmp_VW_file = open(image_dir_VW, 'rb')
        line_VW = tmp_VW_file.readline()
        des_num_tmp = int(float(line_VW.split(',')[0]))
        des_dimension_tmp = int(float(line_VW.split(',')[1]))
        ## line 2, labels
        line_VW = tmp_VW_file.readline()
        data_image_labels = np.zeros((1,des_num_tmp), np.int32)
        for j in range(des_num_tmp):
            data_image_labels[0,j] = int(float(line_VW.split(',')[j]))
        ## line 3, VW
        line_VW = tmp_VW_file.readline()
        data_image_VW = np.zeros((1,des_dimension), np.int32)
        for j in range(des_dimension):
            data_image_VW[0,j] = int(float(line_VW.split(',')[j]))


        ## tmp for descriptors
        des_mat_tmp = np.zeros((des_num_tmp, des_dimension_tmp), np.int32)
        des_tmp_file = open(image_dir_des,'rb')
        reader_des = csv.reader(des_tmp_file, delimiter=',', quoting = csv.QUOTE_NONE)
        row_count = 0
        for row in reader_des:
            for j in range(len(row)):
                des_mat_tmp[row_count, j] = int(float(row[j]))
            row_count += 1
        des_tmp_file.close()
        ## tmp for kpts
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
            kp_tmp.octave = int(float(row[5]))
            kp_tmp.class_id = int(float(row[6]))
            kpts_tmp.append(kp_tmp)
        kpts_tmp_file.close()
        tmp_VW_file.close()

        ## grouping on VW
        data_image_top_des = []
        data_image_top_kpts = []
        print
        for j in range(3):
            print target_image_VW[0, target_image_VW.argsort(axis = 1)[0, (-1-j)]]
        print
        print data_image_labels

        print

        for j in range(how_many_top):
            des_tmp = np.zeros((data_image_VW[0,target_image_VW.argsort(axis = 1)[0,(-1-j)]], des_dimension), np.float32)
            print
            print 'des_tmp.shape', des_tmp.shape
            print
            top_kpts_tmp = []
            count_tmp = 0
            print data_image_labels
            print data_image_VW
            for k in range(des_num_tmp):
                if data_image_labels[0,k] == target_image_VW.argsort(axis = 1)[0, (-1-j)]:
                    ## for descriptors
                    for k_sift in range(des_dimension):
                        ## wrong! it should be the data_image_top_des instead of des_target
                        des_tmp[count_tmp, k_sift] = des_mat_tmp[k,k_sift]
                    ## for kpts
                    kp_tmp = cv2.KeyPoint()
                    kp_tmp.pt = (kpts_tmp[k].pt[0], kpts_tmp[k].pt[1])
                    kp_tmp.size = float(kpts_tmp[k].size)
                    kp_tmp.angle = float(kpts_tmp[k].angle)
                    kp_tmp.response = float(kpts_tmp[k].response)
                    kp_tmp.octave = int(float(kpts_tmp[k].octave))
                    kp_tmp.class_id = int(float(kpts_tmp[k].class_id))
                    top_kpts_tmp.append(kp_tmp)
                    count_tmp += 1
            data_image_top_des.append(des_tmp)
            data_image_top_kpts.append(top_kpts_tmp)

        # for j in range(how_many_top):
        #     print data_image_top_des[j].shape
        #     print len(data_image_top_kpts[j])

        ## now we match on VW
        print
        print 'Match it!!!!!!'
        print 'image ', i
        # print data_image_top_des
        print type(des_target[0,0])

        ## from here. if the VW appears more than 5 time, then we perform the homography on it.




        for j in range(how_many_top):
            ## threshold: 5
            if data_image_top_des[j].shape[0] >= 5:
                print 'round ', j
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                matches = bf.knnMatch(np.uint8(target_image_top_des[j]), trainDescriptors = np.uint8(data_image_top_des[j]), k = 2)
                print len(matches)
                p1, p2, kp_pairs = filter_matches(target_image_top_kpts[j], data_image_top_kpts[j], matches, 0.99)
                if len(kp_pairs) > 0:
                    try:
                        explore_match('find_obj', img_gray,img_tmp,kp_pairs)
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                    except:
                        print 'error!!!'
                else:
                    print 'not enough pairs.'
                    # print np.uint8(target_image_top_des[j])
                    print np.uint8(data_image_top_des[j])
                    print 'len of kp_pairs: ', len(kp_pairs)
