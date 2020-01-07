# -*- coding: utf-8 -*-
import numpy as np 
import copy

from utils import *
from track_param import Track_params

def forward_tracking(track_obj1, track_obj2, track_params, fr_idx2, tracklet_mat, img1, img2):
    new_track_params = copy.deepcopy(track_params)
    new_track_obj1 = track_obj1
    new_track_obj2 = track_obj2
    new_tracklet_mat =tracklet_mat

    if new_track_params.max_track_id == 0:
        new_track_obj1.track_id = np.asarray(list(range(1, new_track_obj1.bbox.shape[0]+1)))
        track_params.max_track_id = new_track_obj1.bbox.shape[0]
        new_track_params.max_track_id = track_params.max_track_id
        new_tracklet_mat.xmin_mat = -1 * np.ones((new_track_obj1.bbox.shape[0], new_track_params.num_fr))
        new_tracklet_mat.ymin_mat = -1 * np.ones((new_track_obj1.bbox.shape[0], new_track_params.num_fr))
        new_tracklet_mat.xmax_mat = -1 * np.ones((new_track_obj1.bbox.shape[0], new_track_params.num_fr))
        new_tracklet_mat.ymax_mat = -1 * np.ones((new_track_obj1.bbox.shape[0], new_track_params.num_fr))
        new_tracklet_mat.color_mat = -1 * np.ones((new_track_obj1.bbox.shape[0], new_track_params.num_fr, 3))
        new_tracklet_mat.class_mat = [[[] for _ in range(new_track_params.num_fr)] for _ in range(new_track_obj1.bbox.shape[0])]
        new_tracklet_mat.det_score_mat = -1 * np.ones((new_track_obj1.bbox.shape[0], new_track_params.num_fr))
        new_tracklet_mat.mask_flag = new_track_obj1.mask_flag
        for i in range(new_track_obj1.bbox.shape[0]):
            new_tracklet_mat.xmin_mat[i, fr_idx2-1] = new_track_obj1.bbox[i, 0]
            new_tracklet_mat.ymin_mat[i, fr_idx2-1] = new_track_obj1.bbox[i, 1]
            new_tracklet_mat.xmax_mat[i, fr_idx2-1] = new_track_obj1.bbox[i, 0] + new_track_obj1.bbox[i, 2] - 1
            new_tracklet_mat.ymax_mat[i, fr_idx2-1] = new_track_obj1.bbox[i, 1] + new_track_obj1.bbox[i, 3] - 1

            bbox_img = img1[new_tracklet_mat.ymin_mat[i, fr_idx2-1].astype(np.int16):new_tracklet_mat.ymax_mat[i, fr_idx2-1].astype(np.int16)+1, new_tracklet_mat.xmin_mat[i, fr_idx2-1].astype(np.int16):new_tracklet_mat.xmax_mat[i, fr_idx2-1].astype(np.int16)+1, :]

            new_tracklet_mat.color_mat[i, fr_idx2-1, 0] = np.mean(bbox_img[:,:,0])
            new_tracklet_mat.color_mat[i, fr_idx2-1, 1] = np.mean(bbox_img[:,:,1])
            new_tracklet_mat.color_mat[i, fr_idx2-1, 2] = np.mean(bbox_img[:,:,2])

            # new_tracklet_mat.class_mat[i, fr_idx2-1] = new_track_obj1.det_class[i]

            new_tracklet_mat.det_score_mat[i, fr_idx2-1] = new_track_obj1.det_score[i]

    # Linear prediction
    track_id1 = new_track_obj1.track_id
    pred_bbox1 = np.zeros((len(track_id1), 4))
    for i in range(len(track_id1)):
        temp_xmin = new_tracklet_mat.xmin_mat[track_id1[i].astype(np.int16)-1,new_tracklet_mat.xmin_mat[track_id1[i].astype(np.int16)-1, :] >= 0]
        temp_xmax = new_tracklet_mat.xmax_mat[track_id1[i].astype(np.int16)-1,new_tracklet_mat.xmax_mat[track_id1[i].astype(np.int16)-1, :] >= 0]
        temp_ymin = new_tracklet_mat.ymin_mat[track_id1[i].astype(np.int16)-1,new_tracklet_mat.ymin_mat[track_id1[i].astype(np.int16)-1, :] >= 0]
        temp_ymax = new_tracklet_mat.ymax_mat[track_id1[i].astype(np.int16)-1,new_tracklet_mat.ymax_mat[track_id1[i].astype(np.int16)-1, :] >= 0]
        temp_list = [temp_xmin, temp_xmax, temp_ymin, temp_ymax]
        temp_bbox = []
        for j in range(4):
            train_t = cnt_idx(new_tracklet_mat.xmin_mat[track_id1[i].astype(np.int16)-1, :] >= 0)
            if len(train_t) > 10:
                temp_t = temp_list[j][-10:]
                train_t = train_t[-10:]
            else:
                temp_t = temp_list[j]
            temp_bbox.append(linearPred(train_t, temp_t, fr_idx2))
        pred_bbox1[i, :] = [temp_bbox[0], temp_bbox[2], temp_bbox[1]-temp_bbox[0]+1, temp_bbox[3]-temp_bbox[2]+1]

    out_idx1 = new_track_obj1.mask_flag < 0.5
    out_idx1 = out_idx1.reshape((-1))
    in_idx1 = new_track_obj1.mask_flag > 0.5
    in_idx1 = in_idx1.reshape((-1))
    out_idx2 = new_track_obj2.mask_flag < 0.5
    out_idx2 = out_idx2.reshape((-1))
    in_idx2 = new_track_obj2.mask_flag > 0.5
    in_idx2 = in_idx2.reshape((-1))
    out_bbox1 = new_track_obj1.bbox[out_idx1, :]
    in_bbox1 = new_track_obj1.bbox[in_idx1, :]
    pred_out_bbox1 = pred_bbox1[out_idx1, :]
    pred_in_bbox1 = pred_bbox1[in_idx1, :]
    out_bbox2 = new_track_obj2.bbox[out_idx2, :]
    in_bbox2 = new_track_obj2.bbox[in_idx2, :]
    N_out_bbox1 = out_bbox1.shape[0]
    N_in_bbox1 = in_bbox1.shape[0]
    N_out_bbox2 = out_bbox2.shape[0]
    N_in_bbox2 = in_bbox2.shape[0]
    out_bbox_color1 = np.zeros((N_out_bbox1, 3))
    in_bbox_color1 = np.zeros((N_in_bbox1, 3))
    out_bbox_color2 = np.zeros((N_out_bbox2, 3))
    in_bbox_color2 = np.zeros((N_in_bbox2, 3))

    for i in range(N_out_bbox1):
        bbox_img = img1[out_bbox1[i,1].astype(np.int16):out_bbox1[i,1].astype(np.int16)+out_bbox1[i,3].astype(np.int16)-1, out_bbox1[i,0].astype(np.int16):out_bbox1[i,0].astype(np.int16)+out_bbox1[i,2].astype(np.int16),:]
        out_bbox_color1[i, 0] = np.mean(bbox_img[:,:,0])
        out_bbox_color1[i, 1] = np.mean(bbox_img[:,:,1])
        out_bbox_color1[i, 2] = np.mean(bbox_img[:,:,2])

    for i in range(N_in_bbox1):
        bbox_img = img1[in_bbox1[i,1].astype(np.int16):in_bbox1[i,1].astype(np.int16)+in_bbox1[i,3].astype(np.int16)-1, in_bbox1[i,0].astype(np.int16):in_bbox1[i,0].astype(np.int16)+in_bbox1[i,2].astype(np.int16),:]
        in_bbox_color1[i, 0] = np.mean(bbox_img[:,:,0])
        in_bbox_color1[i, 1] = np.mean(bbox_img[:,:,1])
        in_bbox_color1[i, 2] = np.mean(bbox_img[:,:,2])

    for i in range(N_out_bbox2):
        bbox_img = img1[out_bbox2[i,1].astype(np.int16):out_bbox2[i,1].astype(np.int16)+out_bbox2[i,3].astype(np.int16)-1, out_bbox2[i,0].astype(np.int16):out_bbox2[i,0].astype(np.int16)+out_bbox2[i,2].astype(np.int16),:]
        out_bbox_color2[i, 0] = np.mean(bbox_img[:,:,0])
        out_bbox_color2[i, 1] = np.mean(bbox_img[:,:,1])
        out_bbox_color2[i, 2] = np.mean(bbox_img[:,:,2])

    for i in range(N_in_bbox2):
        bbox_img = img1[in_bbox2[i,1].astype(np.int16):in_bbox2[i,1].astype(np.int16)+in_bbox2[i,3].astype(np.int16)-1, in_bbox2[i,0].astype(np.int16):in_bbox2[i,0].astype(np.int16)+in_bbox2[i,2].astype(np.int16),:]
        in_bbox_color2[i, 0] = np.mean(bbox_img[:,:,0])
        in_bbox_color2[i, 1] = np.mean(bbox_img[:,:,1])
        in_bbox_color2[i, 2] = np.mean(bbox_img[:,:,2])

    D_r_out = pdist(out_bbox_color1[:,0], out_bbox_color2[:,0])
    D_g_out = pdist(out_bbox_color1[:,1], out_bbox_color2[:,1])
    D_b_out = pdist(out_bbox_color1[:,2], out_bbox_color2[:,2])
    temp_idx = np.max((D_r_out, D_g_out, D_b_out))
    if temp_idx < 6:
        D_max_out = D_r_out
    elif temp_idx < 12:
        D_max_out = D_g_out
    else:
        D_max_out = D_b_out


    D_r_in = pdist(in_bbox_color1[:,0].reshape((-1,1)), in_bbox_color2[:,0].reshape((-1,1)))
    D_g_in = pdist(in_bbox_color1[:,1].reshape((-1,1)), in_bbox_color2[:,1].reshape((-1,1)))
    D_b_in = pdist(in_bbox_color1[:,2].reshape((-1,1)), in_bbox_color2[:,2].reshape((-1,1)))
    temp_idx = np.argmax((D_r_in, D_g_in, D_b_in))
    if temp_idx < 6:
        D_max_in = D_r_in
    elif temp_idx < 12:
        D_max_in = D_g_in
    else:
        D_max_in = D_b_in


    mask_out = D_max_out < new_track_params.color_thresh
    mask_in = D_max_in < new_track_params.color_thresh

    track_id2 = np.zeros((1, N_out_bbox2+N_in_bbox2))

    out_bbox1_idx, out_bbox2_idx, out_overlap_mat = bboxAssociate(pred_out_bbox1, out_bbox2, new_track_params.over_lap_thresh2, new_track_params.lb_thresh, mask_out)
    new_track_obj1.out_overlap_mat = out_overlap_mat
    if track_id2[out_idx2[out_bbox2_idx]]:
        track_id2[out_idx2[out_bbox2_idx]] = track_id1[out_idx1[out_bbox1_idx]]

    in_bbox1_idx, in_bbox2_idx, in_overlap_mat = bboxAssociate(pred_in_bbox1, in_bbox2, new_track_params.over_lap_thresh1, new_track_params.lb_thresh, mask_in)
    new_track_obj1.in_overlap_mat = in_overlap_mat
    track_id2 = track_id2.reshape(-1,)
    track_id2[in_bbox2_idx] = track_id1[in_bbox1_idx]
    # track_id2 = track_id2.reshape((1,-1))

    for i in range(N_out_bbox2+N_in_bbox2):
        if track_id2[i] == 0:
            track_id2[i] = new_track_params.max_track_id+1
            new_track_params.max_track_id = new_track_params.max_track_id+1
        
    new_track_obj2.track_id = track_id2

    if new_track_params.max_track_id > track_params.max_track_id:
        new_tracklet_mat.xmin_mat = np.vstack((new_tracklet_mat.xmin_mat, -1*np.ones((new_track_params.max_track_id-track_params.max_track_id, new_tracklet_mat.xmin_mat.shape[1]))))
        new_tracklet_mat.ymin_mat = np.vstack((new_tracklet_mat.ymin_mat, -1*np.ones((new_track_params.max_track_id-track_params.max_track_id, new_tracklet_mat.ymin_mat.shape[1]))))
        new_tracklet_mat.xmax_mat = np.vstack((new_tracklet_mat.xmax_mat, -1*np.ones((new_track_params.max_track_id-track_params.max_track_id, new_tracklet_mat.xmax_mat.shape[1]))))
        new_tracklet_mat.ymax_mat = np.vstack((new_tracklet_mat.ymax_mat, -1*np.ones((new_track_params.max_track_id-track_params.max_track_id, new_tracklet_mat.ymax_mat.shape[1]))))
        new_tracklet_mat.color_mat = np.vstack((new_tracklet_mat.color_mat, -1*np.ones((new_track_params.max_track_id-track_params.max_track_id, new_tracklet_mat.color_mat.shape[1], 3))))
        # new_tracklet_mat.class_mat = new_tracklet_mat.class_mat+[[[] for _ in range(new_track_params.max_track_id-track_params.max_track_id)] for _ in range(new_track_obj1.det_score_mat.shape[1])]
        new_tracklet_mat.det_score_mat = np.vstack((new_tracklet_mat.det_score_mat, -1*np.ones((new_track_params.max_track_id-track_params.max_track_id, new_tracklet_mat.det_score_mat.shape[1]))))

    if np.max(track_id2).astype(np.int16) > new_tracklet_mat.mask_flag.shape[0]:
        temp_mask = np.zeros((1, np.max(track_id2).astype(np.int16))).reshape((-1,))
        for i in range(new_tracklet_mat.mask_flag.shape[0]):
            if new_tracklet_mat.mask_flag[i]:
                temp_mask[i]  = 1
    for i in range(N_out_bbox2+N_in_bbox2):
        new_tracklet_mat.xmin_mat[track_id2[i].astype(np.int16)-1, fr_idx2] = new_track_obj2.bbox[i, 0]
        new_tracklet_mat.ymin_mat[track_id2[i].astype(np.int16)-1, fr_idx2] = new_track_obj2.bbox[i, 1]
        new_tracklet_mat.xmax_mat[track_id2[i].astype(np.int16)-1, fr_idx2] = new_track_obj2.bbox[i, 0] + new_track_obj2.bbox[i, 2] - 1
        new_tracklet_mat.ymax_mat[track_id2[i].astype(np.int16)-1, fr_idx2] = new_track_obj2.bbox[i, 1] + new_track_obj2.bbox[i, 3] - 1

        bbox_img = img2[new_tracklet_mat.ymin_mat[track_id2[i].astype(np.int16)-1, fr_idx2].astype(np.int16):new_tracklet_mat.ymax_mat[track_id2[i].astype(np.int16)-1, fr_idx2].astype(np.int16), new_tracklet_mat.xmin_mat[track_id2[i].astype(np.int16)-1, fr_idx2].astype(np.int16):new_tracklet_mat.xmax_mat[track_id2[i].astype(np.int16)-1, fr_idx2].astype(np.int16), :]
        new_tracklet_mat.color_mat[track_id2[i].astype(np.int16)-1, fr_idx2, 0] = np.mean(bbox_img[:,:,0])
        new_tracklet_mat.color_mat[track_id2[i].astype(np.int16)-1, fr_idx2, 1] = np.mean(bbox_img[:,:,1])
        new_tracklet_mat.color_mat[track_id2[i].astype(np.int16)-1, fr_idx2, 2] = np.mean(bbox_img[:,:,2])

        # new_tracklet_mat.class_mat[track_id2[i].astype(np.int16)-1][fr_idx2] = new_track_obj2.det_class[i]
        new_tracklet_mat.det_score_mat[track_id2[i].astype(np.int16)-1, fr_idx2] = new_track_obj2.det_score[i]
        if np.max(track_id2).astype(np.int16) > new_tracklet_mat.mask_flag.shape[0]:
            temp_mask[track_id2[i].astype(np.int16)-1] = new_track_obj2.mask_flag[i]
        else:
            new_tracklet_mat.mask_flag[track_id2[i].astype(np.int16)-1] = new_track_obj2.mask_flag[i]
    if np.max(track_id2).astype(np.int16) > new_tracklet_mat.mask_flag.shape[0]:
        new_tracklet_mat.mask_flag = temp_mask



    return new_track_obj1, new_track_obj2, new_tracklet_mat, new_track_params


    

