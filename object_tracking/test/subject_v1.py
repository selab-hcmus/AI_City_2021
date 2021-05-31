"""Find subject track in list of tracking results
"""
import os, sys, pickle, json
import os.path as osp 
import pandas as pd 
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import shutil
import logging

from dataset.data_manager import (
    test_track_map, train_track_map, DATA_DIR,
    TRAIN_TRACK_ORDER_JSON, TEST_TRACK_ORDER_JSON,
)
from utils import (
    AverageMeter, xyxy_to_xywh, xywh_to_xyxy, json_save, json_load
)
from object_tracking.test.test_utils import (
    a_substract_b, is_miss_frame, get_miss_frame_tracks,
    SAVE_DIR
)
from object_tracking.tools import visualize, visualize_subject
from object_tracking.deep_sort.iou_matching import iou
from object_tracking.utils import TRAIN_TRACKING_RESULT, TEST_TRACKING_RESULT


ID_TO_CHECK = [str(i) for i in [5, 6, 9, 84]]
IOU_ACCEPT_THRES = 0.2 #not use yet

def get_top_longest_tracks(vid_data: dict, top_k: int=5):
    list_lens = []
    track_map = vid_data['track_map']
    for track_id in track_map:
        list_lens.append((track_id, len(track_map[track_id]['frame_order'])))
        pass
        
    sorted(list_lens, key=lambda val: val[1])
    if top_k > len(list_lens):
        top_k = len(list_lens)
    
    return [i[0] for i in list_lens[:top_k]]

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

def get_top_nearest_tracks(key: str, vid_data: dict, label_boxes: list, top_k: int=5):  
    list_lens = []
    track_map = vid_data['track_map']
    COMPARE_RANGE = 30
    frame_idx_to_compare = 0

    for track_id in track_map:
        track_frame_order = track_map[track_id]['frame_order']

        distance_meter = AverageMeter()
        track_range = min(len(track_frame_order), COMPARE_RANGE)
        for frame_idx_to_compare in range(track_range):
            track_box = track_map[track_id]['boxes'][frame_idx_to_compare]
            frame_idx = track_frame_order[frame_idx_to_compare]
            cor_subject_box = label_boxes[frame_idx]
            
            cur_distance = calculate_distance(track_box, cor_subject_box)
            distance_meter.update(cur_distance)

        list_lens.append( (track_id, len(track_frame_order), distance_meter.avg) )

    list_lens = sorted(list_lens, key=lambda val: val[-1])
    if key == '349':
        print(list_lens)
    # sorted(list_lens, key=lambda val: val[1])
    if top_k > len(list_lens):
        top_k = len(list_lens)
    
    return [i[0] for i in list_lens[:top_k]]

label_dict = json_load(TRAIN_TRACK_ORDER_JSON)
def check_track_subject(save_dir: str=SAVE_DIR, json_dir: str=TRAIN_TRACKING_RESULT,
                        visualize=False, json=True):
    ambiguous_track = {}
    list_keys = []
    for fname in os.listdir(json_dir):
        key = fname.split('.')[0]
        list_keys.append(key)

    if len(list_keys) == 0:
        print(f'Found no tracking result in {json_dir}')
        return

    print(f'Start finding main subject track')
    for i in tqdm(list_keys):
        save_path = osp.join(save_dir, f'{i}.avi')
        vid_json = osp.join(json_dir, f'{i}.json')
        vid_data = json_load(vid_json)

        subject_boxes = label_dict[i]['boxes']
        subject_boxes = [xywh_to_xyxy(box) for box in subject_boxes]

        # top_longest_track_ids = get_top_longest_tracks(vid_data, top_k=5)
        top_longest_track_ids = get_top_nearest_tracks(i, vid_data, subject_boxes, top_k=1)
        if len(top_longest_track_ids) > 1:
            # print(f'{i}: {len(top_longest_track_ids)}')
            ambiguous_track[i] = len(top_longest_track_ids)
            pass

        if visualize and (not osp.isfile(save_path)):
            visualize_subject(vid_data, top_longest_track_ids, DATA_DIR, save_path, subject_boxes)
    
    if json:
        fail_path = osp.join(save_dir, 'ambiguous_subjects.json')
        json_save(ambiguous_track, fail_path)
    pass

if __name__ == '__main__':
    exp_id = 'deepsort_v4'
    sub_id = 'v1'
    vid_save_dir = osp.join(SAVE_DIR, exp_id, f'video_sub-{sub_id}')
    os.makedirs(vid_save_dir, exist_ok=True)

    check_track_subject(
        save_dir=vid_save_dir, json_dir=osp.join(SAVE_DIR, exp_id, 'json_old'),
        visualize=False, json=False
    )
    pass

