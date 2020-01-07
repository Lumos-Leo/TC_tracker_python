# -*- coding: utf-8 -*-

class Track_params(object):
    def __init__(self, img_size, num_fr=None, const_fr_thresh=None, over_lap_thresh1=None, over_lap_thresh2=None, lb_thresh=None, max_track_id=None, color_thresh=None, det_scope_thresh=None):
        self.img_size = img_size
        self.num_fr = num_fr
        self.const_fr_thresh = const_fr_thresh
        self.over_lap_thresh1 = over_lap_thresh1
        self.over_lap_thresh2 = over_lap_thresh2
        self.lb_thresh = lb_thresh
        self.max_track_id = max_track_id
        self.color_thresh = color_thresh
        self.det_scope_thresh = det_scope_thresh

class Cluster_params(object):
    def __init__(self):
        self.t_dist_thresh = None 