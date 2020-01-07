# -*- coding: utf-8 -*-

from Tc_tracker import tc_tracker
import warnings
warnings.filterwarnings('ignore')

IMG_PATH = 'your/path/to/DETRAC/database/Insight-MVT_Annotation_Test/MVI_39031'
DET_PATH = 'your/path/to/UA_test/EB/MVI_39031_Det_EB.txt'
SEQ_NAME = 'MVI_39031'
ROI_PATH = []
IMG_SAVE_PATH = '/your/path/to/TC_TRACKER/MVI_39031'
RESULT_SAVE_PATH = '/your/path/to/TC_TRACKER/test_result'
VIDEO_SAVE_PATH = '/your/path/to/TC_TRACKER/tracking_video/'

if __name__ == '__main__':
    tc_tracker(IMG_PATH, DET_PATH, ROI_PATH, IMG_SAVE_PATH, SEQ_NAME, RESULT_SAVE_PATH, VIDEO_SAVE_PATH)