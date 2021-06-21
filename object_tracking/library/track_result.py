import cv2 
import os 
import os.path as osp
import numpy as np
from PIL import Image 

from utils.data_manager import DATA_DIR
from classifier.utils import split_data
# from object_tracking.utils import expand_box_area

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
YELLOW = (0, 255, 255)

from utils import create_logger
debug_logger = create_logger(
    '/home/ntphat/projects/AI_City_2021/results/object_tracking/class_color.log', 
    stdout=False
)

def get_valid_coor(x, delta, xmax, xmin=0):
    x_new = x+delta
    x_new = min(xmax, max(xmin, x_new))
    return x_new

def expand_box_area(box: list, W: int, H: int, ratio: float=0.1):
    # xyxy 
    x1, y1, x2, y2 = box 
    w, h = x2-x1, y2-y1
    exp_w = w*ratio
    exp_h = h*ratio

    new_x1 = int(get_valid_coor(x1, -exp_w, W-1, 1))
    new_x2 = int(get_valid_coor(x2, exp_w, W-1, 1))
    new_y1 = int(get_valid_coor(y1, -exp_h, H-1, 1))
    new_y2 = int(get_valid_coor(y2, exp_h, H-1, 1))

    return [new_x1, new_y1, new_x2, new_y2]


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
            box = self.boxes[i]
            box = [int(j) for j in box]

            # # Method 2
            # img = Image.open(fpath)
            # x_0, y_0, x_1, y_1 = box
            # cv_box = img.crop((x_0, y_0, x_1, y_1))
            
            # Method 1
            cv_img = cv2.imread(fpath)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            H, W, C = cv_img.shape

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
        if isinstance(boxes_to_use[0], np.ndarray):
            first_shape = boxes_to_use[0].shape
            last_shape = boxes_to_use[-1].shape
        else:
            first_shape = boxes_to_use[0].size #w, h
            last_shape = boxes_to_use[-1].size #w, h
            
        S_first = first_shape[0]*first_shape[1]
        S_last = last_shape[0]*last_shape[1]


        if abs(S_first - S_last) < AREA_THRES:
            weights[0] = 3
            weights[-1] = 3
        elif S_last > S_first:
            weights[-1] = 3
        else:
            weights[0] = 3
        
        box_names, self.final_vehicle, _, _, self.flag_thres = \
            classifier_manager.get_veh_predictions(boxes_to_use, thres, weights)
        pass 

    def set_col_class(self, list_frames, classifier_manager, thres=0.3):
        AREA_THRES = 10 #pixels
        EXPAND_BOX_RATIO = 0.1
        N = len(self.boxes)
        split = 10
        ids_to_use = split_data(range(N), min(split, N))
        # ids_to_use = sorted(list(set([0, N//6, 2*N//6, 3*N//6, 4*N//6, 5*N//6, N-1])))
        # ids_to_use = list(set([N//10, 2*N//10, 3*N//10, 4*N//10, 5*N//10, 6*N//10, 7*N//10, 8*N//10, 9*N//10]))
        weights = np.ones(len(ids_to_use))
        
        # Set boxes used for classification
        boxes_to_use = self.get_boxes_data(list_frames, ids_to_use, EXPAND_BOX_RATIO)
        # Set box weight
        # first_shape = boxes_to_use[0].shape
        # last_shape = boxes_to_use[-1].shape
        # S_first = first_shape[0]*first_shape[1]
        # S_last = last_shape[0]*last_shape[1]
        # if abs(S_first - S_last) < AREA_THRES:
        #     weights[0] = 3
        #     weights[-1] = 3
        # elif S_last > S_first:
        #     weights[-1] = 3
        # else:
        #     weights[0] = 3
        
        box_names, self.final_color, preds, final_pred, thres_final_pred = \
            classifier_manager.get_col_predictions(boxes_to_use, thres, weights)
        
        str_pred = '['
        for val in final_pred:
            str_pred += f'{val:.4f}, '
        str_pred += ']'
        debug_logger.info(f'[{self.track_id}_{thres}]: Before thres= {str_pred}, After thres= {thres_final_pred}')
        # if self.is_subject:
        #     debug_logger.info(f'Subject preds: {preds}')
        
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
