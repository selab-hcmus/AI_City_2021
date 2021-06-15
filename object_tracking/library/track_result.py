"""Class to handle tracking result

Input: pkl file created by convert_video_track (object_tracking/tools/convert_track.py)
"""
import cv2 
import os 
import os.path as osp
import numpy as np

from utils.data_manager import DATA_DIR
from utils import dict_save, dict_load

from object_tracking.utils import (
    expand_box_area
)


class TrackResult(object):
    def __init__(self, track_id: str, track_info: dict) -> None:
        super().__init__()
        self.track_id = track_id
        self.frame_order = track_info['frame_order']
        self.first_frame = track_info['frame_order'][0]
        self.last_frame = track_info['frame_order'][-1]
        self.boxes = track_info['boxes']
        # self.vehicle_type = track_info['vehicle_type']
        # self.color = track_info['color']
        self.cv_boxes = None
        self.features = []
        self.features = track_info.get('features')
        self.is_subject=False
        self.final_vehicle = None
        self.final_color = None
        self.flag_thres = None
        pass
    pass

    def get_boxes_data(self, list_frames: list, list_ids=None, expand_ratio=None):
        cv_boxes = []
        if list_ids is None:
            list_ids = range(len(self.frame_order))

        for i in list_ids:
            order = self.frame_order[i]
            fpath = osp.join(DATA_DIR, list_frames[order])
            cv_img = cv2.imread(fpath)
            H, W, C = cv_img.shape
            box = self.boxes[i] #xyxy
            box = [int(j) for j in box]
            
            if expand_ratio is not None:
                box = expand_box_area(box, W, H, expand_ratio)
            
            cv_box = cv_img[box[1]:box[3], box[0]: box[2], :]
            cv_boxes.append(cv_box)

        return cv_boxes
        
    def set_veh_class(self, list_frames, classifier_manager, thres=0.7):
        AREA_THRES = 10 #pixels
        EXPAND_BOX_RATIO = 0.1
        N = len(self.boxes)
        ids_to_use = sorted(list(set([0, N//6, 2*N//6, 3*N//6, 4*N//6, 5*N//6, N-1])))
        # ids_to_use = list(set([N//10, 2*N//10, 3*N//10, 4*N//10, 5*N//10, 6*N//10, 7*N//10, 8*N//10, 9*N//10]))
        weights = np.ones(len(ids_to_use))
        
        # Set boxes used for classification
        boxes_to_use = self.get_boxes_data(list_frames, ids_to_use, EXPAND_BOX_RATIO)

        # Set box weight
        first_shape = boxes_to_use[0].shape
        last_shape = boxes_to_use[-1].shape
        S_first = first_shape[0]*first_shape[1]
        S_last = last_shape[0]*last_shape[1]


        if abs(S_first - S_last) < AREA_THRES:
            weights[0] = 3
            weights[-1] = 3
        elif S_last > S_first:
            weights[-1] = 3
        else:
            weights[0] = 3
        
        box_names, self.final_vehicle, _, _, self.flag_thres = classifier_manager.get_veh_predictions(boxes_to_use, thres, weights)
        pass 

    def set_col_class(self, list_frames, classifier_manager, thres=0.4):
        AREA_THRES = 10 #pixels
        EXPAND_BOX_RATIO = 0.1
        N = len(self.boxes)
        ids_to_use = sorted(list(set([0, N//6, 2*N//6, 3*N//6, 4*N//6, 5*N//6, N-1])))
        # ids_to_use = list(set([N//10, 2*N//10, 3*N//10, 4*N//10, 5*N//10, 6*N//10, 7*N//10, 8*N//10, 9*N//10]))
        weights = np.ones(len(ids_to_use))
        
        # Set boxes used for classification
        boxes_to_use = self.get_boxes_data(list_frames, ids_to_use, EXPAND_BOX_RATIO)

        # Set box weight
        first_shape = boxes_to_use[0].shape
        last_shape = boxes_to_use[-1].shape
        S_first = first_shape[0]*first_shape[1]
        S_last = last_shape[0]*last_shape[1]

        if abs(S_first - S_last) < AREA_THRES:
            weights[0] = 3
            weights[-1] = 3
        elif S_last > S_first:
            weights[-1] = 3
        else:
            weights[0] = 3
        
        box_names, self.final_color, _, _ = classifier_manager.get_col_predictions(boxes_to_use, thres, weights)
        pass 

    def to_json(self, is_feat=True):
        res = {}
        res['is_subject'] = self.is_subject
        res['vehicle'] = self.final_vehicle
        res['color'] = self.final_color
        res['frame_order'] = self.frame_order
        res['boxes'] = self.boxes
        res['features'] = self.features
        if is_feat == False:
            res['features'] = []
        return res

    def get_final_classname(self):
        res = None
        if self.final_color is None and self.final_vehicle is not None:
            res = self.final_vehicle
        
        elif self.final_color is not None and self.final_vehicle is not None:
            res = f'{self.final_color}-{self.final_vehicle}'
        return res

