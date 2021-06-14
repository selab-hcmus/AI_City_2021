import numpy as np
import os 
import os.path as osp

from utils.data_manager import RESULT_DIR
## GLOBAL CONSTANT
SAVE_DIR = osp.join(RESULT_DIR, 'object_tracking_exp') #'/content/AI_City_2021/results/object_tracking_exp'
os.makedirs(SAVE_DIR, exist_ok=True)
ID_TO_COMPARE = [5, 6, 9, 20, 34, 40, 64, 84, 182, 188, 239, 310, 339, 349, 410, 436, 476]


## HELPER FUNCTIONS

def calculate_iou(box_a, box_b):
    # x y x y
    bb1 = {'x1': box_a[0], 'y1': box_a[1], 'x2': box_a[2], 'y2': box_a[3]}
    bb2 = {'x1': box_b[0], 'y1': box_b[1], 'x2': box_b[2], 'y2': box_b[3]}

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou

def calculate_distance(box_a, box_b):
    # xyxy
    xa_center = (box_a[0] + box_a[2])/2
    ya_center = (box_a[1] + box_a[3])/2

    xb_center = (box_b[0] + box_b[2])/2
    yb_center = (box_b[1] + box_b[3])/2

    return np.sqrt((xa_center-xb_center)**2 + (ya_center - yb_center)**2)

def is_box_in_box(box_a: list, box_b: list):
    # xyxy
    bb1 = {'x1': box_a[0], 'y1': box_a[1], 'x2': box_a[2], 'y2': box_a[3]}
    bb2 = {'x1': box_b[0], 'y1': box_b[1], 'x2': box_b[2], 'y2': box_b[3]}

    if (bb2['x1'] > bb1['x1'] and bb2['y1'] > bb1['y1']) and \
        (bb2['x2'] < bb1['x2'] and bb2['y2'] < bb1['y2']):
        return True

    return False

def a_substract_b(list_a: list, list_b: list):
    return [ i for i in list_a if i not in list_b]
    
def is_miss_frame(track_data: dict):
    prev = int(track_data['frame_order'][0])
    for frame_order in track_data['frame_order'][1:]:
        order = int(frame_order)
        if order - prev != 1:
            return True
        prev = order
        pass
    return False

def get_miss_frame_tracks(vid_data: dict):
    fail_tracks = []
    for track_id in vid_data['track_map']:
        if is_miss_frame(vid_data['track_map'][track_id]):
            fail_tracks.append(track_id)
        pass
    return fail_tracks