# -*- coding: utf-8 -*-
from track_param import Track_params

class Tracklet_mat(object):
    def __init__(self):
        self.xmin_mat = None
        self.ymin_mat = None
        self.xmax_mat = None
        self.ymax_mat = None
        self.color_mat = None
        self.class_mat = None
        self.det_score_mat = None
        self.mask_flag = None

class Track_obj(object):
    def __init__(self, track_id=[], bbox=[], det_class=[], det_score=[], mask_flag=[]):
        self.track_id = track_id
        self.bbox = bbox
        self.det_class = det_class
        self.det_score = det_score
        self.mask_flag = mask_flag


class Track_struct(object):
    def __init__(self,img_size, num_fr, const_fr_thresh, over_lap_thresh1, over_lap_thresh2, lb_thresh, max_track_id, color_thresh, det_scope_thresh):
        self.track_params = Track_params(img_size, num_fr, const_fr_thresh, over_lap_thresh1, over_lap_thresh2, lb_thresh, max_track_id, color_thresh, det_scope_thresh)
        self.tracklet_mat = Tracklet_mat()
        self.track_obj = []