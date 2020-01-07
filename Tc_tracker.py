# -*- coding: utf-8 -*-
import os
import cv2 as cv 
import time

import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from utils import *
from Parameter import Param
from track_param import Track_params
from track_struct import Track_struct, Track_obj
from tracking import forward_tracking


def tc_tracker(img_folder, det_path, ROI_path, save_path, seq_name, result_save_path, video_save_path):
    param = Param()
    rand_color = np.random.rand(1500, 3)
    img_list = sorted(list(filter(lambda x : x if x.endswith('jpg') else None, os.listdir(img_folder))), key=lambda x:x[3:8])
    # img_list = img_list[:500]
    if not ROI_path:
        temp_img = cv.imread(os.path.join(img_folder, img_list[0]))
        mask = np.ones((temp_img.shape[0], temp_img.shape[1]))
        size = (temp_img.shape[1], temp_img.shape[0])
    else:
        mask = cv.imread(ROI_path)
        size = mask.shape

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(video_save_path+'output.avi', fourcc, 30, size)
    # Read detection file
    print('Read detection file...')
    detections = np.genfromtxt(det_path, delimiter=',')
    M = np.zeros((detections.shape[0], 10))
    for i in range(7):
        if i != 6:
            M[:, i] = np.round(detections[:, i]).astype(np.int16)
        else:
            M[:, i] = detections[:, i]

    M[:, 2] = M[:, 2] + 1
    M[:, 3] = M[:, 3] + 1
    track_p = Track_params(mask.shape, len(img_list), 10, param.IOU_thresh, 0.8, 0.3, 0, param.color_thresh, param.det_score_thresh)
    for i in range(M.shape[0]):
        M[i, 4] = min(M[i ,4], track_p.img_size[1]-M[i, 2]+1)
        M[i, 5] = min(M[i, 5], track_p.img_size[0]-M[i, 3]+1)

    track_s = Track_struct(mask.shape, len(img_list), 10, param.IOU_thresh, 0.8, 0.3, 0, param.color_thresh, param.det_score_thresh)
    for i in range(1, track_p.num_fr+1):
        print('frame {}'.format(i))
        temp_M = M[M[:, 0] == i]
        idx = temp_M[:, 6] > track_p.det_scope_thresh
        det_bbox = temp_M[idx, 2:6]
        temp_score = temp_M[idx, 6]
        idx = idx[idx]

        _, choose_idx = mergeBBOx(det_bbox, 0.8, temp_score)

        idx = idx[choose_idx]

        mask_flag = np.ones((idx.shape[0], 1))
        left_pts = np.round([det_bbox[idx, 0], det_bbox[idx, 1] + det_bbox[idx, 3] - 1])
        right_pts = np.round([det_bbox[idx, 0] + det_bbox[idx, 2] - 1, det_bbox[idx, 1] + det_bbox[idx, 3] - 1])

        right_idx = (right_pts[0, :] - 1) * track_p.img_size[0] + right_pts[1, :]
        left_idx = (left_pts[0, :] - 1) * track_p.img_size[0] + left_pts[1, :]

        right_idx[right_idx < 0] = 1
        left_idx[left_idx < 0] = 1

        mask = mask.reshape([-1])
        a = mask[right_idx.astype(np.int)] > 0.5 
        b = mask[left_idx.astype(np.int)] > 0.5
        mask_flag = mask_flag[a+b]
        mask_flag = mask_flag[temp_score > track_p.det_scope_thresh]

        if not idx.all():
            continue
        track_s.track_obj.append(Track_obj(bbox=det_bbox[idx], det_score=temp_score[idx], mask_flag=mask_flag))
    print('Finishing reading...')
    
    tic = time.time()
    # forward tracking
    for i in range(1, track_p.num_fr):
            
        if i == 1:
            img1 = cv.imread(os.path.join(img_folder, img_list[i-1]))/255
            b, g, r = cv.split(img1)
            img1 = cv.merge([r, g, b])
        img2 = cv.imread(os.path.join(img_folder, img_list[i]))/255
        b, g, r = cv.split(img2)
        img2 = cv.merge([r, g, b])
        track_s.track_obj[i-1], track_s.track_obj[i], track_s.tracklet_mat, track_s.track_params = forward_tracking(
            track_s.track_obj[i-1], track_s.track_obj[i], track_s.track_params, i, track_s.tracklet_mat,img1, img2)

        print('forward {}'.format(i))
        img1 = img2

    iters = 10
    track_s.tracklet_mat = preprocessing(track_s.tracklet_mat, 5)
    for i in range(iters):
        track_s.tracklet_mat, flag, _ = trackletClusterInit(track_s.tracklet_mat, param)
        print('iter_n = {}'.format(i))
        if flag == 1:
            break 

    track_s.prev_tracklet_mat, track_s.tracklet_mat = postProcessing(track_s.tracklet_mat, track_s.track_params)

    sigma = 8
    remove_idx = []
    N_tracklet = track_s.tracklet_mat.xmin_mat.shape[0]
    xmin_reg = [[] for _ in range(N_tracklet)]
    ymin_reg = [[] for _ in range(N_tracklet)]
    xmax_reg = [[] for _ in range(N_tracklet)]
    ymax_reg = [[] for _ in range(N_tracklet)]
    for i in range(N_tracklet):
        det_idx = np.where(track_s.tracklet_mat.xmin_mat[i, :] >= 0)[0]
        print('forward_n = {}'.format(i))
        if len(det_idx) < track_s.track_params.const_fr_thresh:
            remove_idx.append(i)
            continue
        # kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
        kernel = Matern(nu=2.5, length_scale_bounds=(1000,1000))
        model_xmin = gpr(kernel=kernel)
        model_ymin = gpr(kernel=kernel)
        model_xmax = gpr(kernel=kernel)
        model_ymax = gpr(kernel=kernel)
        xmin_reg[i] = model_xmin.fit(det_idx.reshape(-1,1)+1, track_s.tracklet_mat.xmin_mat[i, det_idx].reshape(-1,1))
        ymin_reg[i] = model_ymin.fit(det_idx.reshape(-1,1)+1, track_s.tracklet_mat.ymin_mat[i, det_idx].reshape(-1,1))
        xmax_reg[i] = model_xmax.fit(det_idx.reshape(-1,1)+1, track_s.tracklet_mat.xmax_mat[i, det_idx].reshape(-1,1))
        ymax_reg[i] = model_ymax.fit(det_idx.reshape(-1,1)+1, track_s.tracklet_mat.ymax_mat[i, det_idx].reshape(-1,1))

        t_min = np.min(det_idx)
        t_max = np.max(det_idx)

        track_s.tracklet_mat.xmin_mat[i, t_min:t_max+1] = xmin_reg[i].predict(np.arange(t_min, t_max+1).reshape(-1,1)+1).reshape((-1,))
        track_s.tracklet_mat.ymin_mat[i, t_min:t_max+1] = ymin_reg[i].predict(np.arange(t_min, t_max+1).reshape(-1,1)+1).reshape((-1,))
        track_s.tracklet_mat.xmax_mat[i, t_min:t_max+1] = xmax_reg[i].predict(np.arange(t_min, t_max+1).reshape(-1,1)+1).reshape((-1,))
        track_s.tracklet_mat.ymax_mat[i, t_min:t_max+1] = ymax_reg[i].predict(np.arange(t_min, t_max+1).reshape(-1,1)+1).reshape((-1,))

    track_s.tracklet_mat.xmin_mat = delete_matrix(track_s.tracklet_mat.xmin_mat, remove_idx, 2)
    track_s.tracklet_mat.ymin_mat = delete_matrix(track_s.tracklet_mat.ymin_mat, remove_idx, 2)
    track_s.tracklet_mat.xmax_mat = delete_matrix(track_s.tracklet_mat.xmax_mat, remove_idx, 2)
    track_s.tracklet_mat.ymax_mat = delete_matrix(track_s.tracklet_mat.ymax_mat, remove_idx, 2)
    track_s.tracklet_mat.color_mat = delete_matrix(track_s.tracklet_mat.color_mat, remove_idx, 3)
    # track_s.tracklet_mat.class_mat = delete_matrix(track_s.tracklet_mat.class_mat, remove_idx, 2)
    track_s.tracklet_mat.det_score_mat = delete_matrix(track_s.tracklet_mat.det_score_mat, remove_idx, 2)
    print('finish tracking ....')
    toc = time.time()


    for t in range(track_p.num_fr):
        frame = cv.imread(os.path.join(img_folder, img_list[t]))
        for i in range(track_s.tracklet_mat.xmin_mat.shape[0]):
            if track_s.tracklet_mat.xmin_mat[i, t] == -1:
                continue
            
            x_min = track_s.tracklet_mat.xmin_mat[i, t]
            y_min = track_s.tracklet_mat.ymin_mat[i, t]
            x_max = track_s.tracklet_mat.xmax_mat[i, t]
            y_max = track_s.tracklet_mat.ymax_mat[i, t]


            font = cv.FONT_HERSHEY_DUPLEX
            frame = cv.rectangle(frame, (int(x_min), int(y_min)),
                        (int(x_max), int(y_max)),
                        (0, 255, 0), 3)
            frame = cv.putText(frame, str(t), (0,50), font, 1, (0,0,255), 1)
            frame = cv.putText(frame, str(i), (int(x_min), int(y_min)-6), font, 2, (0,0,255), 2)

        cv.imwrite( save_path + '/img{:0>6}.jpg'.format(t), frame)
        out.write(frame)
        # cv.imshow('video_name', frame)
        # cv.waitKey(40)
    

    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # frame_ = cv.imread(os.path.join(save_path, list(os.listdir(save_path))[0]))
    # size = (frame_.shape[1], frame_.shape[0])
    # print(size)
    # out = cv.VideoWriter(video_save_path+'output.avi', fourcc, 20, size)
    # print(sorted(list(filter(lambda x : x if x.endswith('jpg') else None, os.listdir(save_path))), key=lambda x:x[3:]))
    # for x in sorted(list(filter(lambda x : x if x.endswith('jpg') else None, os.listdir(save_path))), key=lambda x:x[3:]):
        # frame_ = cv.imread(os.path.join(save_path, x))
        # out.write(frame_)
        # cv.imshow('video_name', frame_)
        # cv.waitKey(0)

    out.release()

    Speed = len(img_list) / (toc - tic)
    if seq_name:
        writetxt(seq_name, track_s, result_save_path, Speed)

    return 0

    

    
