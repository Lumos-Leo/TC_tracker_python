# -*- coding: utf-8 -*-
import copy

import numpy as np 
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from track_param import Cluster_params

def writetxt(seq_name, track_s, save_path, speed):
    X = np.transpose(track_s.tracklet_mat.xmin_mat)
    Y = np.transpose(track_s.tracklet_mat.ymin_mat)
    W = np.transpose(track_s.tracklet_mat.xmax_mat) - X 
    H = np.transpose(track_s.tracklet_mat.ymax_mat) - Y 

    W[W < 0] = 0
    H[H < 0] = 0
    Y[Y < 0] = 0
    X[X < 0] = 0

    fileID = save_path + '/' + seq_name + '_LX.txt'
    with open(fileID, 'w') as f:
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                f.write('{}'.format(X[r, c]))
                if c != X.shape[1]-1:
                    f.write(',',)
                else:
                    f.write('\n')

    fileID = save_path + '/' + seq_name + '_LY.txt'
    with open(fileID, 'w') as f:
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                f.write('{}'.format(Y[r, c]))
                if c != X.shape[1]-1:
                    f.write(',')
                else:
                    f.write('\n')

    fileID = save_path + '/' + seq_name + '_W.txt'
    with open(fileID, 'w') as f:
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                f.write('{}'.format(W[r, c]))
                if c != X.shape[1]-1:
                    f.write(',')
                else:
                    f.write('\n')

    fileID = save_path + '/' + seq_name + '_H.txt'
    with open(fileID, 'w') as f:
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                f.write('{}'.format(H[r, c]))
                if c != X.shape[1]-1:
                    f.write(',')
                else:
                    f.write('\n')

    fileID = save_path + '/' + seq_name + '_speed.txt'
    with open(fileID, 'w') as f:
        f.write('{}'.format(speed))





def max_mat(A):
    if A.shape[0] > 1 and A.shape[1] > 1:
        t_idx = np.argmax(A)
        max_v = np.max(A)
        idx = np.array([t_idx//A.shape[1], t_idx%A.shape[1]])
    elif A.shape[0] == 1:
        max_v = np.max(A)
        idx = np.array([0, np.argmax(A)])
    elif A.shape[1] == 1:
        max_v = np.max(A)
        idx = np.array([np.argmax(A), 0])

    return max_v, idx 

def find(a, b):
    return a[:, 6] > b

def overlap(boxA, boxB):
    boxA = boxA.reshape(-1, 4)
    boxB = boxB.reshape(-1, 4)
    N1 = boxA.shape[0]
    N2 = boxB.shape[0]
    _, xmin1 = np.meshgrid(np.arange(1,N2+1), boxA[:,0])
    xmin2, _ = np.meshgrid(boxB[:, 0], np.arange(1,N1+1))
    _, ymin1 = np.meshgrid(np.arange(1,N2+1), boxA[:,1])
    ymin2, _ = np.meshgrid(boxB[:, 1], np.arange(1,N1+1))
    _, xmax1 = np.meshgrid(np.arange(1,N2+1), boxA[:,0]+boxA[:,2]-1)
    xmax2, _ = np.meshgrid(boxB[:, 0]+boxB[:, 2]-1, np.arange(1,N1+1))
    _, ymax1 = np.meshgrid(np.arange(1,N2+1), boxA[:,1]+boxA[:,3]-1)
    ymax2, _ = np.meshgrid(boxB[:, 1]+boxB[:, 3]-1, np.arange(1,N1+1))

    temp_xmin1 = xmin1.reshape((-1,1))
    temp_xmin2 = xmin2.reshape((-1,1))
    xmin = np.zeros(temp_xmin1.shape)
    for i in range(len(temp_xmin1)):
        xmin[i] = max(temp_xmin1[i], temp_xmin2[i])
    xmin = xmin.reshape(xmin1.shape)

    temp_ymin1 = ymin1.reshape((-1,1))
    temp_ymin2 = ymin2.reshape((-1,1))
    ymin = np.zeros(temp_ymin1.shape)
    for i in range(len(temp_ymin1)):
        ymin[i] = max(temp_ymin1[i], temp_ymin2[i])
    ymin = ymin.reshape(ymin1.shape)

    temp_xmax1 = xmax1.reshape((-1,1))
    temp_xmax2 = xmax2.reshape((-1,1))
    xmax = np.zeros(temp_xmax1.shape)
    for i in range(len(temp_xmax1)):
        xmax[i] = min(temp_xmax1[i], temp_xmax2[i])
    xmax = xmax.reshape(xmax1.shape)

    temp_ymax1 = ymax1.reshape((-1,1))
    temp_ymax2 = ymax2.reshape((-1,1))
    ymax = np.zeros(temp_ymax1.shape)
    for i in range(len(temp_ymax1)):
        ymax[i] = min(temp_ymax1[i], temp_ymax2[i])
    ymax = ymax.reshape(ymax1.shape)


    mask = np.bitwise_and(xmax > xmin, ymax > ymin)
    ratio_mat = np.zeros((N1,N2))
    overlap_area = np.zeros((N1, N2))
    overlap_area[mask] = (xmax[mask] - xmin[mask]) * (ymax[mask] - ymin[mask])

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    ratio_mat[mask] = overlap_area[mask] / (area1[mask] + area2[mask] - overlap_area[mask])
    return ratio_mat, overlap_area


def mergeBBOx(bbox, thre, score):
    cand_idx = np.ones((bbox.shape[0], 1), dtype=np.int32)
    for n1 in range(bbox.shape[0]-1):
        for n2 in range(n1+1, bbox.shape[0]):
            if cand_idx[n1] == 0 or cand_idx[n2] == 0:
                continue
            r, overlap_area = overlap(bbox[n1, :], bbox[n2, :])
            r1 = overlap_area / (bbox[n1, 2] * bbox[n1, 3])
            r2 = overlap_area / (bbox[n2, 2] * bbox[n2, 3])
            s1 = score[n1]
            s2 = score[n2]

            if r1 > thre or r2 > thre:
                if s1 > s2:
                    cand_idx[n2] = 0
                else:
                    cand_idx[n1] = 0
    idx = cand_idx[:] == 1
    idx = np.reshape(idx, idx.shape[0])
    # temp = []
    # for i in cand_idx:
    #     cnt = 0
    #     if i:
    #         temp.append(bbox[cnt,:])
    #         cnt += 1
    # update_bbox = np.asarray(temp)
    update_bbox = bbox[idx, :]
    return update_bbox, idx

def cnt_idx(a):
    out = []
    cnt = 1
    for i in a:
        if i:
            out.append(cnt)
            cnt += 1
        else:
            cnt += 1
    return np.array(out).reshape((-1,1))


def linearPred(train_t, train_x, t):
    if len(train_x) <= 2:
        pred_x = train_x[-1]
        return pred_x

    A = np.hstack((train_t, np.ones((len(train_t), 1))))
    b = train_x.reshape((-1,1))
    p = np.linalg.pinv(A).dot(b) 
    pred_x = p[0] * (t+1) + p[1]
    return pred_x

def pdist(x, y):
    m = x.shape[0]
    n = y.shape[0]
    if m:
        xx = np.sum(x*x, axis=1).reshape((m,-1))
    else:
        xx = np.sum(x*x)
    if n:
        yy = np.sum(np.transpose(y)*np.transpose(y), axis=0).reshape((-1,n))
    else:
        yy = np.sum(np.transpose(y)*np.transpose(y))
    xx_final = np.copy(xx)
    yy_final = np.copy(yy)
    if n > 0:
        for _ in range(n-1):
            xx_final = np.hstack((xx_final, xx))

    if m > 0:
        for _ in range(m-1):
            yy_final = np.vstack((yy_final, yy))
    
    if m == 0 and n == 0:
        return np.array([10000])
    elif m == 0:
        return xx_final
    elif n == 0:
        return yy_final
    else:
        return np.sqrt(xx_final + yy_final - 2 * np.matmul(x, np.transpose(y)))

def bboxAssociate(det_bbox, gt_bbox, overlap_thresh, lb_thresh, mask):
    if not len(gt_bbox) or not len(det_bbox):
        det_idx = []
        gt_idx = []
        final_overlap_mat = []
        return det_idx, gt_idx, final_overlap_mat
    
    N1 = det_bbox.shape[0]
    N2 = gt_bbox.shape[0]
    overlap_mat,_ = overlap(det_bbox, gt_bbox)
    if mask.any():
        overlap_mat[mask==0] = 0

    final_overlap_mat = np.copy(overlap_mat)

    gt_idx = []
    det_idx = []
    while 1:
        max_v, idx = max_mat(overlap_mat)
        if max_v < overlap_thresh:
            break

        overlap_idx = np.sum(overlap_mat[idx[0], :]>lb_thresh)
        if overlap_idx == 1:
            det_idx.append(idx[0])
            gt_idx.append(idx[1])

        overlap_mat[idx[0], :] = 0
        overlap_mat[:, idx[1]] = 0

    return det_idx, gt_idx,  final_overlap_mat

def preprocessing(tracklet_mat, len_thresh):
    new_tracklet_mat = tracklet_mat 
    N_tracklet = tracklet_mat.xmin_mat.shape[0]
    remove_idx = []
    for i in range(1, N_tracklet+1):
        t = np.sum(tracklet_mat.xmin_mat[i-1, :] >= 0)
        if t < len_thresh:
            remove_idx.append(i)

    for i in range(len(remove_idx)):
        new_tracklet_mat.mask_flag[remove_idx[i]-1] = 0
    return new_tracklet_mat

def getTrackInterval(tracklet_mat):
    N_tracklet = tracklet_mat.xmin_mat.shape[0]
    track_interval = np.zeros((N_tracklet, 2))
    for i in range(tracklet_mat.xmin_mat.shape[1]):
        if i == 0:
            tracklet_mat_xmin_mat_temp = tracklet_mat.xmin_mat[:,i]
        else:
            tracklet_mat_xmin_mat_temp = np.hstack((tracklet_mat_xmin_mat_temp, tracklet_mat.xmin_mat[:,i]))
    cand_idx = np.where(tracklet_mat_xmin_mat_temp >= 0)
    min_mask = np.inf  * np.ones((tracklet_mat.xmin_mat.shape[0] * tracklet_mat.xmin_mat.shape[1])).reshape((-1,))
    cand_idx = np.array(cand_idx).reshape((-1,))
    min_mask[cand_idx] = cand_idx
    for i in range(tracklet_mat.xmin_mat.shape[1]):
        if i == 0:
            temp_mask = min_mask[i*tracklet_mat.xmin_mat.shape[0]:(i+1)*tracklet_mat.xmin_mat.shape[0]].reshape((-1, 1))
        else:
            temp_mask = np.hstack((temp_mask, min_mask[i*tracklet_mat.xmin_mat.shape[0]:(i+1)*tracklet_mat.xmin_mat.shape[0]].reshape((-1, 1))))
    min_mask = temp_mask
    min_v = []

    for i in range(min_mask.shape[0]):
        min_v.append(np.min(min_mask[i,:]))
        track_interval[i,0] = np.argmin(min_mask[i,:])
    min_v = np.array(min_v)
    track_interval[min_v == np.inf, 0] = -1
    max_mask = -1 * np.ones(tracklet_mat.xmin_mat.shape).reshape((-1,))
    max_mask[cand_idx] = cand_idx
    for i in range(tracklet_mat.xmin_mat.shape[1]):
        if i == 0:
            temp_mask = max_mask[i*tracklet_mat.xmin_mat.shape[0]:(i+1)*tracklet_mat.xmin_mat.shape[0]].reshape((-1, 1))
        else:
            temp_mask = np.hstack((temp_mask, max_mask[i*tracklet_mat.xmin_mat.shape[0]:(i+1)*tracklet_mat.xmin_mat.shape[0]].reshape((-1, 1))))

    max_mask = temp_mask
    max_v = []

    for i in range(max_mask.shape[0]):
        max_v.append(np.max(max_mask[i,:]))
        track_interval[i,1] = np.argmax(max_mask[i,:])
    max_v = np.array(max_v)
    track_interval[min_v < 0, 1] = -1
    return track_interval

def getNeighborTrack(track_interval, t_dist_thresh, intersect_ratio):
    N_tracklet = track_interval.shape[0]
    neighbor_idx  = [[] for _ in range(N_tracklet)]
    for i in range(N_tracklet):
        cand_idx = np.where((np.bitwise_and(track_interval[i, 0] - track_interval[:, 1] < t_dist_thresh, track_interval[:, 0] - track_interval[i, 1] < t_dist_thresh)))[0]
        if not len(cand_idx):
            continue
        remove_idx = []
        vec_idx1 = np.arange(track_interval[i, 0], track_interval[i, 1]+1)
        for k in range(cand_idx.reshape(1,-1).shape[1]):
            vec_idx2 = np.arange(track_interval[cand_idx[k], 0], track_interval[cand_idx[k], 1]+1)
            vec_idx3 = np.intersect1d(vec_idx1, vec_idx2)
            if (len(vec_idx3)/min(len(vec_idx1), len(vec_idx2))) > intersect_ratio:
                remove_idx.append(k)

        remove_idx = np.array(remove_idx).reshape((-1,))
        cand_idx = np.delete(cand_idx, remove_idx)
        if not len(cand_idx):
            continue
        neighbor_idx[i].append(cand_idx)
    return neighbor_idx

def bboxToPoint(tracklet_mat):
    new_tracklet_mat = copy.deepcopy(tracklet_mat)
    new_tracklet_mat.det_x = 0.5 * (tracklet_mat.xmin_mat + tracklet_mat.xmax_mat) + 1
    new_tracklet_mat.det_y = 0.5 * (tracklet_mat.ymax_mat + tracklet_mat.ymax_mat) + 1
    new_tracklet_mat.det_x[new_tracklet_mat.det_x < 0] = -1
    new_tracklet_mat.det_y[new_tracklet_mat.det_y < 0] = -1
    return new_tracklet_mat 

def overlapCheck(track_interval1, track_interval2):
    track_interval1 = clear_list(track_interval1)
    track_interval2 = clear_list(track_interval2)
    t_min = max(track_interval1[0], track_interval2[0])
    t_max = min(track_interval1[1], track_interval2[1])
    if t_min > t_max:
        overlap_ratio = 0
        return overlap_ratio
    else:
        min_len = min(track_interval1[1] - track_interval1[0] + 1, track_interval2[1] - track_interval2[0] + 1)
        overlap_ratio = (t_max - t_min) / min_len
    return overlap_ratio

def temp_comb(t):
    if len(t) > 1:
        if isinstance(t, list):
            temp_comb = t[0]
            for i in range(1,len(t)):
                # if len(temp_comb.shape) != len(t[i].shape):
                #     a = 1
                if isinstance(t[i], list):
                    t[i] = t[i][0]
                temp_comb = np.hstack((temp_comb, t[i]))
            t = temp_comb
            return t
        else:
            return t
    elif len(t) == 1:
        if isinstance(t, list):
            return t[0]
        else:
            return t
    else:
        return t

def combCost(track_set, tracklet_mat, cluster_params, appearance_cost=None):
    track_test = copy.deepcopy(track_set)
    track_set = temp_comb(track_set)
    if len(track_set):
        N_tracklet = len(track_set)
    else:
        N_tracklet = 0
    track_interval = tracklet_mat.track_interval
    track_set = track_set.reshape((-1, ))
    sort_idx = np.argsort(track_interval[track_set, 1], axis = 0)
    split_cost = 1

    t = []
    test_t = []
    test_x = []
    test_y = []
    test_w = []
    test_h = []
    det_x = []
    det_y = []
    det_w = []
    det_h = []
    train_t = []
    train_x = []
    train_y = []
    train_w = []
    train_h = []
    end_t = []
    piece_x =[]
    piece_y = []
    len_test = 4
    if N_tracklet <= 1:
        color_flag = 0
    else:
        color_flag = 1
        diff_mean_color = np.zeros((3, N_tracklet-1))

    for i in range(N_tracklet):
        track_id = track_set[sort_idx[i].astype(np.int16)]
        temp_t = np.arange(track_interval[track_id, 0].astype(np.int), track_interval[track_id, 1].astype(np.int)+1)
        if color_flag == 1:
            if i == 0:
                temp_color = tracklet_mat.color_mat[track_id, temp_t, :]
                temp_color = temp_color[np.newaxis, :]
                if len(temp_t) > cluster_params.color_sample_size:
                    end_color = temp_color[0, -cluster_params.color_sample_size-1:, :]
                    end_color = end_color[np.newaxis, :]
                else:
                    end_color = temp_color
            elif i < N_tracklet-1:
                temp_color = tracklet_mat.color_mat[track_id, temp_t, :]
                temp_color = temp_color[np.newaxis, :]
                if len(temp_t) > cluster_params.color_sample_size:
                    start_color = temp_color[0, :cluster_params.color_sample_size, :]
                    start_color = start_color[np.newaxis, :]
                else:
                    start_color = temp_color
                diff_mean_color[0, i-1] = np.abs(np.mean(start_color[:,0]) - np.mean(end_color[:,0]))
                diff_mean_color[1, i-1] = np.abs(np.mean(start_color[:,1]) - np.mean(end_color[:,1]))
                diff_mean_color[2, i-1] = np.abs(np.mean(start_color[:,2]) - np.mean(end_color[:,2]))
                if len(temp_t) > cluster_params.color_sample_size:
                    end_color = temp_color[0,-cluster_params.color_sample_size-1:, :]
                    end_color = end_color[np.newaxis, :]
                else:
                    end_color = temp_color
            else:
                temp_color = tracklet_mat.color_mat[track_id, temp_t, :]
                temp_color = temp_color[np.newaxis, :]
                if len(temp_t) > cluster_params.color_sample_size:
                    start_color = temp_color[0, :cluster_params.color_sample_size, :]
                    start_color = start_color[np.newaxis, :]
                else:
                    start_color = temp_color
                diff_mean_color[0, i-1] = np.abs(np.mean(start_color[0,:,0]) - np.mean(end_color[0,:,0]))
                diff_mean_color[1, i-1] = np.abs(np.mean(start_color[0,:,1]) - np.mean(end_color[0,:,1]))
                diff_mean_color[2, i-1] = np.abs(np.mean(start_color[0,:,2]) - np.mean(end_color[0,:,2]))

        temp_test_t1 = np.arange(temp_t[0], min(temp_t[-1], temp_t[0]+len_test)+1)
        temp_test_t2 = np.arange(max(temp_t[-1]-len_test, temp_t[0]), temp_t[-1]+1)

        if i != 0:
            test_x.append(tracklet_mat.det_x[track_id, temp_test_t1])
            test_y.append(tracklet_mat.det_y[track_id, temp_test_t1])
            test_w.append(tracklet_mat.xmax_mat[track_id, temp_test_t1] - tracklet_mat.xmin_mat[track_id, temp_test_t1] + 1)
            test_h.append(tracklet_mat.ymax_mat[track_id, temp_test_t1] - tracklet_mat.ymin_mat[track_id, temp_test_t1] + 1)
            test_t.append(temp_test_t1)
            end_t.append(temp_t[0])

        if i !=  N_tracklet-1:
            test_x.append(tracklet_mat.det_x[track_id, temp_test_t2])
            test_y.append(tracklet_mat.det_y[track_id, temp_test_t2])
            test_w.append(tracklet_mat.xmax_mat[track_id, temp_test_t2] - tracklet_mat.xmin_mat[track_id, temp_test_t2] + 1)
            test_h.append(tracklet_mat.ymax_mat[track_id, temp_test_t2] - tracklet_mat.ymin_mat[track_id, temp_test_t2] + 1)
            test_t.append(temp_test_t2)
            end_t.append(temp_t[-1])

        det_x.append(tracklet_mat.det_x[track_id, temp_t])
        det_y.append(tracklet_mat.det_y[track_id, temp_t])
        det_w.append(tracklet_mat.xmax_mat[track_id, temp_t] - tracklet_mat.xmin_mat[track_id, temp_t] + 1)
        det_h.append(tracklet_mat.ymax_mat[track_id, temp_t] - tracklet_mat.ymin_mat[track_id, temp_t] + 1)
        t.append(temp_t)
        
        temp_t1 = temp_t[:min(len(temp_t), cluster_params.track_len)]
        temp_t2 = temp_t[-min(len(temp_t), cluster_params.track_len)-1:]
        train_t.append(np.hstack((temp_t1, temp_t2)))
        train_x.append(np.hstack((tracklet_mat.det_x[track_id, temp_t1], tracklet_mat.det_x[track_id, temp_t2])))
        train_y.append(np.hstack((tracklet_mat.det_y[track_id, temp_t1], tracklet_mat.det_y[track_id, temp_t2])))
        train_w.append(
            np.hstack((tracklet_mat.xmax_mat[track_id, temp_t1] - tracklet_mat.xmin_mat[track_id, temp_t1] + 1, tracklet_mat.xmax_mat[track_id, temp_t2] - tracklet_mat.xmin_mat[track_id, temp_t2] + 1
            )))
        train_h.append(
            np.hstack((tracklet_mat.ymax_mat[track_id, temp_t1] - tracklet_mat.ymin_mat[track_id, temp_t1] + 1,tracklet_mat.ymax_mat[track_id, temp_t2] - tracklet_mat.ymin_mat[track_id, temp_t2] + 1 
            )))

    grad_cost = 0
    t = temp_comb(t)
    test_t = temp_comb(test_t)
    test_x = temp_comb(test_x)
    test_y = temp_comb(test_y)
    test_w = temp_comb(test_w)
    test_h = temp_comb(test_h)
    det_x = temp_comb(det_x)
    det_y = temp_comb(det_y)
    det_w = temp_comb(det_w)
    det_h = temp_comb(det_h)
    train_t = temp_comb(train_t)
    train_x = temp_comb(train_x)
    train_y = temp_comb(train_y)
    train_w = temp_comb(train_w)
    train_h = temp_comb(train_h)
    end_t = temp_comb(end_t)

    if len(np.unique(t)) < cluster_params.len_tracklet_thresh:
        reg_cost = cluster_params.small_track_cost
        grad_cost = 0
    else:
        t, idx = np.unique(t, return_index = True)
        det_x = det_x.reshape((-1, ))[idx]
        det_y = det_y.reshape((-1, ))[idx]
        det_w = det_w.reshape((-1, ))[idx]
        det_h = det_h.reshape((-1, ))[idx]

        train_t, idx = np.unique(train_t, return_index = True)
        train_x = train_x.reshape((-1, ))[idx]
        train_y = train_y.reshape((-1, ))[idx]
        train_w = train_w.reshape((-1, ))[idx]
        train_h = train_h.reshape((-1, ))[idx]

        det_size = np.sqrt(np.square(train_w) + np.square(train_h))
        # eng = matlab.engine.start_matlab()
        # model_x = eng.predict(eng.fitrgp(matlab.double(train_t.reshape((-1,1)).tolist()), matlab.double(train_x.reshape((-1,1)).tolist()), 'Basis', 'linear', 'FitMethod', 'exact', 'PredictMethod', 'exact', 'Sigma', matlab.double([cluster_params.sigma]), 'ConstantSigma', matlab.logical([1]), 'KernelFunction', 'matern52', 'KernelParameters', matlab.double([1000,1000])), matlab.double(t.reshape((-1,1)).tolist()))

        kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))

        # kernel = Matern(nu=2.5, length_scale_bounds=(100,100))
        model_x = gpr(kernel=kernel)
        model_x.fit(train_t.reshape((-1,1))+1, train_x.reshape((-1,1)))
        model_y = gpr(kernel=kernel)
        model_y.fit(train_t.reshape((-1,1))+1, train_y.reshape((-1,1)))
        model_bbox = gpr(kernel=kernel)
        model_bbox.fit(train_t.reshape((-1,1))+1, det_size.reshape((-1,1)))
        # model_x = gpr(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        # model_x.fit(train_t.reshape((-1,1))+1, train_x.reshape((-1,1)))
        # model_y = gpr(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        # model_y.fit(train_t.reshape((-1,1))+1, train_y.reshape((-1,1)))
        # model_bbox = gpr(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        # model_bbox.fit(train_t.reshape((-1,1))+1, det_size.reshape((-1,1)))

        if len(test_t):
            test_t, test_idx = np.unique(test_t, return_index=True)
            test_x = test_x[test_idx]
            test_y = test_y[test_idx]
            test_w = test_w[test_idx]
            test_h = test_h[test_idx]
            pred_test_x = model_x.predict(test_t.reshape((-1,1))+1)
            pred_test_y = model_y.predict(test_t.reshape((-1,1))+1)
            test_size = model_bbox.predict(test_t.reshape((-1,1))+1)
            err = np.sum(np.sqrt(np.square((pred_test_x-test_x.reshape((-1,1))))+np.square((pred_test_y-test_y.reshape((-1,1)))))/test_size)
        else:
            err = 0

        reg_cost = err 

        pred_x = model_x.predict(t.reshape((-1,1))+1)
        pred_y = model_y.predict(t.reshape((-1,1))+1)
        smooth_size = model_bbox.predict(t.reshape((-1,1))+1)
        t_min = np.min(t)
        t_max = np.max(t)
        t_t = np.arange(t_min, t_max+1)

        pred_x_t = np.interp(t_t, t, pred_x.reshape((-1,)))
        pred_y_t = np.interp(t_t, t, pred_y.reshape((-1,)))
        pred_size_t = np.interp(t_t, t, smooth_size.reshape((-1,)))

        ax = pred_x_t[2:] + pred_x_t[:-2] - 2 * pred_x_t[1:-1]
        ay = pred_y_t[2:] + pred_y_t[:-2] - 2 * pred_y_t[1:-1]

        if not len(ax):
            grad_cost = 0
        else:
            acc_err = np.sqrt((ax*ax+ay*ay))/(pred_size_t[1:-1]+1e-6)
            t_interval = t_t[1:-1]
            if len(end_t):
                max_grad = np.zeros((end_t.shape))
            else:
                max_grad = np.array([])
            for i in range(len(end_t)):
                temp_idx = np.where(t_interval==end_t[i])
                min_idx = np.max((1, temp_idx[0]-3))
                max_idx = np.min((len(t_interval), temp_idx[0]+3))
                max_grad[i] = np.max(acc_err[int(min_idx):int(max_idx+1)])
            grad_cost = np.sum(max_grad)

    color_cost = 0
    if color_flag == 1:
        max_diff_color = np.max(diff_mean_color)
        color_cost = np.sum(max_diff_color)
    if appearance_cost:
        color_cost = appearance_cost
    if len(sort_idx[0:-1]):
        track_dist = track_interval[track_set[sort_idx[1:]], 0] - track_interval[track_set[sort_idx[0:-1]], 1]
    else:
        track_dist = np.array([])
    if len(track_dist):
        max_dist = np.max(track_dist)
    else:
        max_dist = np.array([])
    if not max_dist or max_dist <= 0:
        time_cost = 0
    else:
        time_cost = np.power(max_dist, 3)/1e6
    
    f = (split_cost, reg_cost, color_cost, grad_cost, time_cost)
    return f



                



def getAssignCost(track_id, tracklet_mat, track_interval, track_cluster, track_class, neighbor_track_idx, prev_cluster_cost, cluster_params, cluster_flag):
    intersect_ratio_thresh = cluster_params.intersect_ratio_thresh
    track_cluster = clear_tuple(track_cluster)
    cluster1 = track_cluster[int(track_class[track_id])-1]
    new_cluster_cost = np.zeros((2, 5))
    new_cluster_set = [[] for _ in range(2)]
    new_cluster_set[0].append(cluster1)
    mask = np.ones(new_cluster_set[0][0].shape, dtype=bool)
    mask[new_cluster_set[0][0]==track_id] = False 
    new_cluster_set[0][0] = new_cluster_set[0][0][mask]

    if len(new_cluster_set[0][0]):
        new_cluster_cost[0, :] = combCost(new_cluster_set[0], tracklet_mat, cluster_params)

    N_cluster = len(track_cluster)
    if not cluster_flag:
        N_cand = len(track_cluster)
    else:
        N_cand = round(sum(cluster_flag))
    temp_new_cluster_cost = np.inf * np.ones((N_cluster, 5))
    prev_cost_vec = np.zeros((N_cluster, 5))
    for i in range(N_cand):
        if track_class[track_id]-1 == i:
            continue

        neighbor_track = np.intersect1d(neighbor_track_idx[track_id], track_cluster[i])
        if not len(neighbor_track):
            continue

        cluster_size = len(track_cluster[i])
        if cluster_size == 0:
            continue

        for k in range(cluster_size):
            overlap_ratio = overlapCheck(
                track_interval[track_cluster[i][k], :], track_interval[track_id, :]
            )
            if overlap_ratio > intersect_ratio_thresh:
                break

        if overlap_ratio > intersect_ratio_thresh:
            continue

        temp_set = [np.array([track_id]), track_cluster[i]]
        temp_new_cluster_cost[i, :] = combCost(temp_set, tracklet_mat, cluster_params)
        prev_cost_vec[i, :] = prev_cluster_cost[int(track_class[track_id]-1), :] + prev_cluster_cost[i, :]
    
    cost_vec = temp_new_cluster_cost + new_cluster_cost[0, :]
    diff_cost_vec = (cost_vec - prev_cost_vec).dot(np.array([cluster_params.lambda_split, cluster_params.lambda_reg, cluster_params.lambda_color, cluster_params.lambda_grad, cluster_params.lambda_time]).reshape((-1, 1)))
    min_idx = np.argmin(diff_cost_vec)
    cost_vec_temp = cost_vec.reshape((-1,))
    cost = cost_vec_temp[min_idx*cost_vec.shape[1]]
    if cost == np.inf:
        diff_cost = np.inf 
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f
    diff_cost = diff_cost_vec[min_idx]
    f = cost_vec[min_idx, :] - prev_cost_vec[min_idx, :]
    new_cluster_cost[1, :] = temp_new_cluster_cost[min_idx, :]

    change_cluster_idx = [track_class[track_id], min_idx+1]
    new_cluster_set[1] = np.array([track_id, int(track_cluster[min_idx][0])])
    return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

def getSplitCost(track_id, track_cluster, track_class, tracklet_mat, prev_cost, cluster_params):
    new_cluster_cost = np.zeros((2, 5))

    if len(track_cluster[track_class[track_id].astype(np.int16)-1]) == 1:
        cost = np.inf
        diff_cost = np.inf
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []

        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f 
    
    change_cluster_idx = np.array([len(track_cluster)+1, track_class[track_id].astype(np.int16)])
    new_cluster_set = [[] for _ in range(2)]
    new_cluster_set[0].append(track_id)
    remain_tracks = track_cluster[track_class[track_id].astype(np.int16)-1]
    remain_tracks = np.array(remain_tracks)
    mask = np.zeros((remain_tracks.shape), dtype=bool)
    mask[remain_tracks!=track_id] = True
    remain_tracks = remain_tracks[mask]
    new_cluster_set[1].append(remain_tracks)

    new_cluster_cost[0, :] = combCost(np.array(new_cluster_set[0]), tracklet_mat, cluster_params)

    if new_cluster_set[1]:
        new_cluster_cost[1, :] = combCost(np.array(new_cluster_set[1]), tracklet_mat, cluster_params)

    cost_temp = np.sum(new_cluster_cost)
    cost = np.zeros((prev_cost.shape))
    cost[0] = cost_temp
    f = cost - prev_cost
    diff_cost = f.dot(np.array([cluster_params.lambda_split, cluster_params.lambda_reg, cluster_params.lambda_color,cluster_params.lambda_grad, cluster_params.lambda_time]))
    return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f 

def getMergeCost(track_id, tracklet_mat, track_interval, track_cluster, track_class, neighbor_track_idx, prev_cluster_cost, cluster_params):
    intersect_ratio_thresh = cluster_params.intersect_ratio_thresh

    cluster1 = track_cluster[int(track_class[track_id])-1]
    if len(cluster1) == 1:
        cost = np.inf
        diff_cost = np.inf
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

    N_cluster = len(track_cluster)
    new_cluster_cost_vec = np.inf * np.ones((N_cluster, 5))
    prev_cost_vec = np.zeros((N_cluster, 5))
    for i in range(N_cluster):
        if track_class[track_id]-1 == i:
            continue

        neighbor_track = np.intersect1d(neighbor_track_idx[track_id], track_cluster[i])
        if not len(neighbor_track):
            continue

        cluster_size = len(track_cluster[i])
        if cluster_size == 0:
            continue

        for k1 in range(len(cluster1)):
            for k2 in range(cluster_size):
                overlap_ratio = overlapCheck(track_interval[track_cluster[i][k2], :], track_interval[cluster1[k1], :])
                if overlap_ratio > intersect_ratio_thresh:
                    break 
            
            if overlap_ratio > intersect_ratio_thresh:
                break

        if overlap_ratio > intersect_ratio_thresh:
            continue

        new_cluster_cost_vec[i, :] = combCost([cluster1, track_cluster[i]], tracklet_mat, cluster_params)
        prev_cost_vec[i, :] = prev_cluster_cost[track_class[track_id].astype(np.int16)-1, :] + prev_cluster_cost[i,:]
    diff_cost_vec = (new_cluster_cost_vec - prev_cost_vec).dot(np.array([cluster_params.lambda_split, cluster_params.lambda_reg, cluster_params.lambda_color,cluster_params.lambda_grad, cluster_params.lambda_time]))
    min_idx = np.argmin(diff_cost_vec)
    cost = new_cluster_cost_vec[min_idx, :]
    if np.sum(cost == np.inf) == len(cost):
        diff_cost = np.inf
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

    diff_cost = diff_cost_vec[min_idx]
    f = new_cluster_cost_vec[min_idx, :] - prev_cost_vec[min_idx, :]
    new_cluster_cost = np.zeros((2, 5))
    new_cluster_cost[0, :] = cost 

    change_cluster_idx = [track_class[track_id], min_idx]
    new_cluster_set = [[] for _ in range(2)]
    new_cluster_set[0] = [cluster1, track_cluster[min_idx]]
    return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

def clear_tuple(f):
    candi = []
    for x in f:
        if isinstance(x, tuple):
            candi.append(x[0])
        elif isinstance(x, list):
            if x:
                candi.append(np.array(x[0]))
            else:
                candi.append(np.array(x))
        else:
            candi.append(x)
    return candi

def clear_list(f):
    if isinstance(f, list):
        return f[0].reshape((-1, ))
    else:
        return f.reshape((-1, ))

def getSwitchCost(track_id, tracklet_mat, track_interval, track_cluster, track_class, neighbor_track_idx, prev_cluster_cost, cluster_params):
    track_cluster = clear_tuple(track_cluster)
    cluster1 = track_cluster[int(track_class[track_id])-1]
    S1 = []
    S2 = []
    for k in range(len(cluster1)):
        temp_id = cluster1[k]
        if track_interval[temp_id, 1] <= track_interval[track_id, 1]:
            S1.append(temp_id)
        else:
            S2.append(temp_id)

    N_cluster = len(track_cluster)
    cost_vec = np.inf * np.ones((N_cluster, 5))
    prev_cost_vec = np.zeros((N_cluster, 5))
    new_cluster_cost_vec1 = np.inf * np.ones((N_cluster, 5))
    new_cluster_cost_vec2 = np.inf * np.ones((N_cluster, 5))
    track_id_set = [[[] for _ in range(5)] for _ in range(N_cluster)]
    t_max_fr = tracklet_mat.xmin_mat.shape[1]
    for i in range(N_cluster):
        stop_flag = 1
        cluster_size = len(track_cluster[i])
        if cluster_size == 0:
            continue 
        for k in range(cluster_size):
            if np.isin(track_cluster[i][k], neighbor_track_idx[track_id]):
                stop_flag = 0
                break
        if stop_flag == 1:
            continue

        cand_track_interval = track_interval[track_cluster[i], :].astype(np.int16)
        t_min = np.min(cand_track_interval[:, 0]).astype(np.int16)
        t_max = np.max(cand_track_interval[:, 1]).astype(np.int16)
        t_check = -1 * np.ones((t_max_fr))

        for k in range(cluster_size):
            t_check[np.arange(cand_track_interval[k, 0],cand_track_interval[k, 1]+1).reshape(-1,)] = 1
        if t_check[track_interval[track_id, 1].astype(np.int16)] == 1:
            continue

        S3 = []
        S4 = []

        for k in range(cluster_size):
            temp_id = track_cluster[i][k]
            if track_interval[temp_id, 1] <= track_interval[track_id, 1]:
                S3.append(temp_id)
            else:
                S4.append(temp_id)

        S3 = temp_comb(S3)
        S4 = temp_comb(S4)
        if isinstance(S4, np.int64):
            S4 = np.array([S4])
        if len(S4):
            S_1 = np.hstack((np.array(S1), np.array(S4)))
        else:
            S_1 = np.array(S1)
        if isinstance(S3, np.int64):
            S3 = np.array([S3])
        if len(S3):
            S_2 = np.hstack((np.array(S3), np.array(S2)))
        else:
            S_2 = np.array(S2)

        neighbor_set1 = []
        for k in range(len(S1)):
            neighbor_set1.append(neighbor_track_idx[S1[k]])
        if isinstance(neighbor_set1, list):
            if neighbor_set1:
                neighbor_set1 = neighbor_set1[0]
            else:
                neighbor_set1 = neighbor_set1
        neighbor_set1 = np.unique(neighbor_set1)
        if not len(np.intersect1d(neighbor_set1, S4)):
            continue

        neighbor_set2 = []
        if isinstance(S3, np.int64):
            S3 = np.array([S3])
        for k in range(len(S3)):
            neighbor_set2.append(neighbor_track_idx[S3[k]])

        neighbor_set2 = np.unique(neighbor_set2)

        if not np.intersect1d(neighbor_set2, S2):
            continue

        new_cluster_cost_vec1[i, :] = combCost(S_1, tracklet_mat, cluster_params)
        new_cluster_cost_vec2[i, :] = combCost(S_2, tracklet_mat, cluster_params)

        cost_vec[i, :] = new_cluster_cost_vec1[i, :] + new_cluster_cost_vec2[i, :]

        track_id_set[i] = [[] for _ in range(2)]
        track_id_set[0].append(S_1)
        track_id_set[1].append(S_2)

        prev_cost_vec[i, :] = prev_cluster_cost[track_class[track_id].astype(np.int16)-1, :] + prev_cluster_cost[i, :]

    diff_cost_vec = (cost_vec - prev_cost_vec).dot(np.array([cluster_params.lambda_split, cluster_params.lambda_reg, cluster_params.lambda_color,cluster_params.lambda_grad, cluster_params.lambda_time]))
    min_idx = np.argmin(diff_cost_vec)
    f = cost_vec[min_idx, :] - prev_cost_vec[min_idx, :]
    cost = cost_vec.reshape((-1,))[min_idx*cost_vec.shape[1]]
    if cost == np.inf:
        diff_cost = np.inf
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f
    
    diff_cost = diff_cost_vec[min_idx]
    new_cluster_cost = np.zeros((2, 5))
    new_cluster_cost[0, :] = new_cluster_cost_vec1[min_idx, :]
    new_cluster_cost[1, :] = new_cluster_cost_vec2[min_idx, :]

    change_cluster_idx = np.hstack((track_class[track_id], min_idx))
    new_cluster_set = [[] for _ in range(2)]
    new_cluster_set[0].append(track_id_set[min_idx][0])
    new_cluster_set[1].append(track_id_set[min_idx][1])
    return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

def getBreakCost(track_id, track_cluster, track_class, tracklet_mat, prev_cost, cluster_params):
    new_cluster_cost = np.zeros((2, 5))
    if len(track_cluster[track_class[track_id].astype(np.int16)-1]) <= 2:
        cost = np.inf 
        diff_cost = np.inf
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

    track_ids = track_cluster[track_class[track_id]]
    track_interval = tracklet_mat.track_interval
    after_ids = track_ids[track_interval[track_ids, 1] > track_interval[track_id, 1]]
    if not after_ids:
        cost = np.inf 
        diff_cost = np.inf
        new_cluster_cost = []
        new_cluster_set = []
        change_cluster_idx = []
        f = []
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f
    else:
        before_ids = np.setdiff1d(track_ids, after_ids)
        if len(before_ids) <= 1:
            cost = np.inf 
            diff_cost = np.inf
            new_cluster_cost = []
            new_cluster_set = []
            change_cluster_idx = []
            f = []
            return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f
        
        change_cluster_idx = np.array([len(track_cluster)+1, track_class[track_id]])
        new_cluster_set = [[] for _ in range(2)]
        new_cluster_set[0].append(before_ids)
        remain_tracks = after_ids
        new_cluster_Set[1] = remain_tracks
        new_cluster_cost[0, :] = combCost(new_cluster_set[0], tracklet_mat, cluster_params)
        new_cluster_cost[1, :] = combCost(new_cluster_set[1], tracklet_mat, cluster_params)
        cost = np.sum(new_cluster_cost)
        f = cost - prev_cost
        diff_cost = f.dot(np.array([cluster_params.lambda_split, cluster_params.lambda_reg, cluster_params.lambda_color,cluster_params.lambda_grad, cluster_params.lambda_time]))
        return cost, diff_cost, new_cluster_cost, new_cluster_set, change_cluster_idx, f

def cnt_temp(f):
    cnt = 0
    for i in range(len(f)):
        if len(f[i]):
            print(cnt)
            cnt += 1
    return cnt

def trackletCluster(tracklet_mat, track_interval, track_cluster, track_class, prev_cluster_cost, neighbor_track_idx, cluster_params, f_mat, track_change_set):
    new_track_cluster = copy.deepcopy(track_cluster)
    new_track_class = copy.deepcopy(track_class)
    new_prev_cluster_cost = copy.deepcopy(prev_cluster_cost)
    new_f_mat = copy.deepcopy(f_mat)
    new_track_change_set = copy.deepcopy(track_change_set)
    
    # new_track_class = new_track_class.reshape((-1, 1))

    sort_track_idx = np.argsort(track_interval[:, 1], kind='stable')
    for i in range(0,len(sort_track_idx)):
        if i == 54:
            dd = 1
        track_id = sort_track_idx[i]
        if new_track_class[track_id] < 0:
            continue

        diff_cost = np.zeros((5,1))
        new_C = [[] for _ in range(5)]
        new_set = [[] for _ in range(5)]
        change_idx = [[] for _ in range(5)]
        f = [[] for i in range(5)]

        _,diff_cost[0], new_C[0], new_set[0], change_idx[0], f[0] = getSplitCost(track_id, new_track_cluster, new_track_class, tracklet_mat, new_prev_cluster_cost[int(new_track_class[track_id])-1, :], cluster_params)
        _,diff_cost[1], new_C[1], new_set[1], change_idx[1], f[1] = getAssignCost(
            track_id, tracklet_mat, track_interval, new_track_cluster, new_track_class, neighbor_track_idx, new_prev_cluster_cost, cluster_params, []
        )
        _,diff_cost[2], new_C[2], new_set[2], change_idx[2], f[2] = getMergeCost(
            track_id, tracklet_mat, track_interval, new_track_cluster, new_track_class, neighbor_track_idx, new_prev_cluster_cost, cluster_params
        )
        _,diff_cost[3], new_C[3], new_set[3], change_idx[3], f[3] = getSwitchCost(
            track_id, tracklet_mat, track_interval, new_track_cluster, new_track_class, neighbor_track_idx, new_prev_cluster_cost, cluster_params
        )
        _,diff_cost[4], new_C[4], new_set[4], change_idx[4], f[4] = getBreakCost(
            track_id, new_track_cluster, new_track_class, tracklet_mat, new_prev_cluster_cost[new_track_class[track_id].astype(np.int16)-1, :], cluster_params
        )

        new_track_cluster = clear_tuple(new_track_cluster)
        for k in range(len(f)):
            if diff_cost[k] == np.inf:
                continue

            cnt_id = len(new_track_change_set)
            if cnt_id > new_f_mat.shape[0]:
                temp_f = np.zeros((new_f_mat.shape[0]*2, 6))
                temp_f[:new_f_mat.shape[0]] = new_f_mat
                new_f_mat = temp_f

            temp_f = f[k]
            min_diff = np.min(np.sum(np.abs(new_f_mat[:,:5]-temp_f), axis=1))
            if min_diff < 1e-6:
                continue
            new_track_change_set_temp = [[[] for _ in range(2)] for _ in range(2)]
            if diff_cost[k] < 0:
                new_f_mat[cnt_id, :5] = f[k]
                new_f_mat[cnt_id, 5] = -1
                if change_idx[k][0] > len(new_track_cluster):
                    new_track_change_set_temp[0][0] = []
                else:
                    new_track_change_set_temp[0][0] = new_track_cluster[change_idx[k][0].astype(np.int16)-1]

                if change_idx[k][1] > len(new_track_cluster):
                    new_track_change_set_temp[0][1] = []
                else:
                    new_track_change_set_temp[0][1] = new_track_cluster[change_idx[k][1].astype(np.int16)-1]

                new_track_change_set_temp[1][0] = new_set[k][0][0]
                new_track_change_set_temp[1][1] = new_set[k][1]
                new_track_change_set[cnt_id] =  copy.deepcopy(new_track_change_set_temp)

            if diff_cost[k] > 0:
                new_f_mat[cnt_id, :5] = f[k]
                new_f_mat[cnt_id, 5] = 1
                if change_idx[k][0] > len(new_track_cluster):
                    new_track_change_set_temp[0][0] = []
                else:
                    new_track_change_set_temp[0][0] = new_track_cluster[change_idx[k][0].astype(np.int16)-1]

                if change_idx[k][1] > len(new_track_cluster):
                    new_track_change_set_temp[0][1] = []
                else:
                    new_track_change_set_temp[0][1] = new_track_cluster[change_idx[k][1].astype(np.int16)-1]

                new_track_change_set_temp[1][0] = new_set[k][0][0]
                new_track_change_set_temp[1][1] = new_set[k][1]
                new_track_change_set[cnt_id] =  copy.deepcopy(new_track_change_set_temp)
        min_cost = np.min(diff_cost)
        min_idx = np.argmin(diff_cost)
        if min_cost >= 0:
            continue
        new_track_cluster[change_idx[min_idx][0].astype(np.int16)-1] = new_set[min_idx][0]
        new_track_cluster[change_idx[min_idx][1].astype(np.int16)-1] = new_set[min_idx][1]
        new_prev_cluster_cost[change_idx[min_idx][0].astype(np.int16)-1, :] = new_C[min_idx][0, :]
        new_prev_cluster_cost[change_idx[min_idx][1].astype(np.int16)-1, :] = new_C[min_idx][1, :]

        if isinstance(new_track_cluster[change_idx[min_idx][0].astype(np.int16)-1], list):
            new_track_cluster[change_idx[min_idx][0].astype(np.int16)-1] = new_track_cluster[change_idx[min_idx][0].astype(np.int16)-1][0]
        for k in range(len(new_track_cluster[change_idx[min_idx][0].astype(np.int16)-1])):
            new_track_class[np.array(np.array(new_track_cluster[change_idx[min_idx][0].astype(np.int16)-1])[k])] = change_idx[min_idx][0].astype(np.int16)

        for k in range(len(new_track_cluster[change_idx[min_idx][1].astype(np.int16)-1])):
            new_track_class[new_track_cluster[change_idx[min_idx][1].astype(np.int16)-1][k]] = change_idx[min_idx][1]

    return new_track_cluster, new_track_class, new_prev_cluster_cost, new_f_mat, new_track_change_set


def sameCheck(struct1, struct2):
    if len(struct1) != len(struct2):
        flag = 0
        return flag

    for i in range(len(struct1)):
        temp_flag = np.array_equal(struct1[i], struct2[i])
        if not temp_flag:
            flag = 0
            return flag 
    
    flag = 1
    return flag 


def trackletClusterInit(tracklet_mat, param):
    cluster_params = Cluster_params()
    cluster_params.t_dist_thresh = 20
    cluster_params.lambda_time = param.lambda_time
    cluster_params.intersect_ratio_thresh = 0.2
    cluster_params.len_tracklet_thresh = 2
    cluster_params.lambda_split = param.lambda_split
    cluster_params.small_track_cost = 0.1
    cluster_params.sigma = 8
    cluster_params.track_len = 25
    cluster_params.lambda_reg = param.lambda_reg
    cluster_params.lambda_color = param.lambda_color
    cluster_params.color_sample_size = 5
    cluster_params.lambda_grad = param.lambda_grad

    new_tracklet_mat = copy.deepcopy(tracklet_mat) 

    if not hasattr(new_tracklet_mat, 'cluster_params'):
        new_tracklet_mat.cluster_params = cluster_params

    if not hasattr(new_tracklet_mat, 'track_interval'):
        new_tracklet_mat.track_interval = getTrackInterval(new_tracklet_mat)

    if not hasattr(new_tracklet_mat, 'track_class'):
        new_tracklet_mat.track_class = np.round(np.cumsum(new_tracklet_mat.mask_flag))
        new_tracklet_mat.track_class[new_tracklet_mat.mask_flag<0.5] = -1

    if not hasattr(new_tracklet_mat, 'track_cluster'):
        N_cluster = max(new_tracklet_mat.track_class).astype(np.int16)
        new_tracklet_mat.track_cluster = [[] for i in range(N_cluster)]
        for i in range(N_cluster):
            new_tracklet_mat.track_cluster[i] = np.where(new_tracklet_mat.track_class == i+1)

    if not hasattr(new_tracklet_mat, 'neighbor_track_idx'):
        new_tracklet_mat.neighbor_track_idx = getNeighborTrack(new_tracklet_mat.track_interval, cluster_params.t_dist_thresh, cluster_params.intersect_ratio_thresh)

    if not hasattr(new_tracklet_mat, 'det_x') or not hasattr(new_tracklet_mat, 'det_y'):
        new_tracklet_mat = bboxToPoint(new_tracklet_mat)

    if not hasattr(new_tracklet_mat, 'cluster_cost'):
        N_cluster = max(new_tracklet_mat.track_class).astype(np.int16)
        new_tracklet_mat.cluster_cost = np.zeros((N_cluster, 5))
        new_tracklet_mat.cluster_cost[:, 0] = 1

    if not hasattr(new_tracklet_mat, 'f_mnat'):
        new_tracklet_mat.f_mat = np.zeros((1000, 6))

    if not hasattr(new_tracklet_mat, 'track_change_set'):
        new_tracklet_mat.track_change_set = {}

    prev_track_cluster = copy.deepcopy(new_tracklet_mat.track_cluster) 
    new_tracklet_mat.track_cluster, new_tracklet_mat.track_class, new_tracklet_mat.cluster_cost, new_tracklet_mat.f_mat, new_tracklet_mat.track_change_set = trackletCluster(new_tracklet_mat, new_tracklet_mat.track_interval, new_tracklet_mat.track_cluster, new_tracklet_mat.track_class, new_tracklet_mat.cluster_cost, new_tracklet_mat.neighbor_track_idx, cluster_params, new_tracklet_mat.f_mat, new_tracklet_mat.track_change_set)
    flag = sameCheck(prev_track_cluster, new_tracklet_mat.track_cluster)

    cnt = 0
    for i in range(len(new_tracklet_mat.track_cluster)):
        if len(new_tracklet_mat.track_cluster[i]) >= 1:
            cnt = cnt + 1

    return new_tracklet_mat, flag, cnt

def updateTrackletMat(tracklet_mat):
    new_tracklet_mat = copy.deepcopy(tracklet_mat)
    track_interval = copy.deepcopy(tracklet_mat.track_interval)
    num_cluster = np.sum(tracklet_mat.cluster_flag).astype(np.int16)

    new_xmin_mat = -1 * np.ones((num_cluster, new_tracklet_mat.xmin_mat.shape[1]))
    new_ymin_mat = -1 * np.ones((num_cluster, new_tracklet_mat.xmin_mat.shape[1]))
    new_xmax_mat = -1 * np.ones((num_cluster, new_tracklet_mat.xmin_mat.shape[1]))
    new_ymax_mat = -1 * np.ones((num_cluster, new_tracklet_mat.xmin_mat.shape[1]))
    new_color_mat = -1 * np.ones((num_cluster, new_tracklet_mat.xmin_mat.shape[1], 3))
    new_class_mat = [[[] for _ in range(new_tracklet_mat.xmin_mat.shape[1])]for _ in range(num_cluster)]
    new_det_score_mat = -1 * np.ones((num_cluster, new_tracklet_mat.xmin_mat.shape[1]))

    for i in range(num_cluster):
        for k in range(len(new_tracklet_mat.track_cluster[i])):
            temp_id = new_tracklet_mat.track_cluster[i][k]
            new_xmin_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1] = new_tracklet_mat.xmin_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1]
            new_ymin_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1] = new_tracklet_mat.ymin_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1]
            new_xmax_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1] = new_tracklet_mat.xmax_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1]
            new_ymax_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1] = new_tracklet_mat.ymax_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1]
            new_color_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1, :] = new_tracklet_mat.color_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1, :]
            # new_class_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1] = new_tracklet_mat.class_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1]
            new_det_score_mat[i, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1] = new_tracklet_mat.det_score_mat[temp_id, track_interval[temp_id, 0].astype(np.int16):track_interval[temp_id, 1].astype(np.int16)+1]

    new_tracklet_mat.xmin_mat = new_xmin_mat
    new_tracklet_mat.ymin_mat = new_ymin_mat
    new_tracklet_mat.xmax_mat = new_xmax_mat
    new_tracklet_mat.ymax_mat = new_ymax_mat
    new_tracklet_mat.color_mat = new_color_mat
    new_tracklet_mat.class_mat = new_class_mat
    new_tracklet_mat.det_score_mat = new_det_score_mat

    return new_tracklet_mat





def postProcessing(tracklet_mat, track_params):
    new_tracklet_mat = copy.deepcopy(tracklet_mat)
    N_cluster = len(tracklet_mat.track_cluster)
    remove_idx = []
    for i in range(N_cluster):
        if not len(tracklet_mat.track_cluster[i]):
            remove_idx.append(i)
            continue
        cnt = 0
        temp_ids = np.zeros((len(tracklet_mat.track_cluster[i])))
        for k in range(len(tracklet_mat.track_cluster[i])):
            track_id = tracklet_mat.track_cluster[i][k]
            temp_ids[k] = track_id
            cnt = cnt + tracklet_mat.track_interval[track_id, 1] - tracklet_mat.track_interval[track_id, 0] + 1

        if cnt < track_params.const_fr_thresh:
            remove_idx.append(i)
            new_tracklet_mat.mask_flag[temp_ids.astype(np.int16)] = 0
    mask = np.ones((N_cluster,), dtype=bool)
    mask[np.array(remove_idx).reshape((-1,))] = False
    new_tracklet_mat.track_cluster = list(np.array(new_tracklet_mat.track_cluster)[mask])
    
    for i in range(new_tracklet_mat.cluster_cost.shape[1]):
        if i == 0:
            temp_new_tracklet_mat_cluster_cost = new_tracklet_mat.cluster_cost[:, i]
        else:
            temp_new_tracklet_mat_cluster_cost = np.hstack((temp_new_tracklet_mat_cluster_cost, new_tracklet_mat.cluster_cost[:, i]))

    new_tracklet_mat.cluster_cost = temp_new_tracklet_mat_cluster_cost
    new_tracklet_mat.cluster_cost = np.delete(new_tracklet_mat.cluster_cost, np.array(remove_idx) )
    new_tracklet_mat.cluster_flag  = np.zeros((new_tracklet_mat.track_class.shape))
    new_tracklet_mat.cluster_flag[:len(new_tracklet_mat.track_cluster)] = 1

    new_tracklet_mat.track_class = -1 * np.ones((new_tracklet_mat.track_class.shape))
    N_cluster = len(new_tracklet_mat.track_cluster)

    for i in range(N_cluster):
        for k in range(len(new_tracklet_mat.track_cluster[i])):
            track_id = new_tracklet_mat.track_cluster[i][k]
            new_tracklet_mat.track_class[track_id] = i+1

    for i in range(len(new_tracklet_mat.track_class)):
        if new_tracklet_mat.track_class[i] > 0:
            continue 

        new_tracklet_mat.track_cluster.append(np.array([i]))
        new_tracklet_mat.track_class[i] = len(new_tracklet_mat.track_cluster)

    N_id = np.round(np.sum(new_tracklet_mat.cluster_flag))
    N_cluster = len(new_tracklet_mat.track_cluster)
    N_fr = track_params.num_fr
    N_tracklet = len(new_tracklet_mat.track_class)
    new_track_cluster = copy.deepcopy(new_tracklet_mat.track_cluster)
    new_track_class = copy.deepcopy(new_tracklet_mat.track_class)
    cluster_params = copy.deepcopy(new_tracklet_mat.cluster_params)
    track_interval = copy.deepcopy(new_tracklet_mat.track_interval)
    neighbor_track_idx = copy.deepcopy(new_tracklet_mat.neighbor_track_idx)

    new_tracklet_mat.track_cluster[int(N_id): ] = []
    new_tracklet_mat2 = updateTrackletMat(new_tracklet_mat)

    return new_tracklet_mat, new_tracklet_mat2


def delete_matrix(f, remove_idx_mask, dimension = 2):
    mask = np.ones(f.shape, dtype = bool)
    if dimension == 2:
        w, h = f.shape
        mask[remove_idx_mask, :] = False
        return f[mask].reshape((w-len(remove_idx_mask), h))
    else:
        w, h, d = f.shape
        mask[remove_idx_mask, :, :] = False
    return f[mask].reshape((w-len(remove_idx_mask), h, d)) 
