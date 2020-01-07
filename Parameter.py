# -*- coding: utf-8 -*-
# Author:LiAo

# parameter setting
class Param(object):
    def __init__(self):
        self.det_score_thresh = 0.1
        self.IOU_thresh = 0.5
        self.color_thresh = 0.15
        self.lambda_time = 25
        self.lambda_split = 0.35
        self.lambda_reg = 0.2
        self.lambda_color = 0.25
        self.lambda_grad = 8
        
