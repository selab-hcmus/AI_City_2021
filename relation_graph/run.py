import pickle 
import os 
import os.path as osp 
import numpy as np 
import cv2


TRAIN_TRACKING_RESULT = '../object_tracking/results/annotate_train'
TEST_TRACKING_RESULT = '../object_tracking/results/annotate_test'

SAVE_DIR = './data'

def concat_track(track_data):
    track_map = {}
    list_frames = list(track_data.keys())

    for frame_name in list_frames:
        for track_ele in track_data[frame_name]:
            
            

            pass
        pass
    pass

