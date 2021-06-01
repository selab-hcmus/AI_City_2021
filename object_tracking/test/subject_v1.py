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
    a_substract_b, is_miss_frame, get_miss_frame_tracks, calculate_iou, calculate_distance,
    SAVE_DIR
)
from object_tracking.test.evaluate_subject import evaluate, TRAIN_SVO_IDS
from object_tracking.tools import visualize, visualize_subject
from object_tracking.deep_sort.iou_matching import iou
from object_tracking.utils import TRAIN_TRACKING_RESULT, TEST_TRACKING_RESULT

from object_tracking.library.track_result import TrackResult
from object_tracking.library import VideoResult

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

def get_top_nearest_tracks(key: str, vid_data: VideoResult, label_boxes: list, top_k: int=5):  
    list_lens = []
    track_map = vid_data.track_map
    COMPARE_RANGE = 30

    for track_id in track_map:
        track_frame_order = track_map[track_id].frame_order

        distance_meter = AverageMeter()
        iou_meter = AverageMeter()
        track_range = min(len(track_frame_order), COMPARE_RANGE)

        for i in range(track_range):
            track_box = track_map[track_id].boxes[i]
            frame_idx = track_frame_order[i]
            cor_subject_box = label_boxes[frame_idx]
            
            cur_distance = calculate_distance(track_box, cor_subject_box)
            cur_iou = calculate_iou(track_box, cor_subject_box)
            distance_meter.update(cur_distance)
            iou_meter.update(cur_iou)

        list_lens.append( (track_id, len(track_frame_order), iou_meter.avg, distance_meter.avg) )

    list_lens = sorted(list_lens, key=lambda val: val[-1])
    if top_k > len(list_lens):
        top_k = len(list_lens)
    list_lens = list_lens[:top_k]
    final_results = [val for val in list_lens if val[2] >= IOU_ACCEPT_THRES]

    return [i[0] for i in final_results]

label_dict = json_load(TRAIN_TRACK_ORDER_JSON)
def check_track_subject(save_dir: str=SAVE_DIR, json_dir: str=TRAIN_TRACKING_RESULT,
                        visualize=False, json=True):
    ambiguous_track = {}
    list_keys = []
    for fname in os.listdir(json_dir):
        key = fname.split('.')[0]
        list_keys.append(key)
    
    list_keys = list(set(list_keys))
    if len(list_keys) == 0:
        print(f'Found no tracking result in {json_dir}')
        return

    print(f'Start finding main subject track')
    total_score_dict = {}
    for i in tqdm(list_keys):
        save_path = osp.join(save_dir, f'{i}.avi')
        vid_json = osp.join(json_dir, f'{i}.json')
        vid_data = VideoResult(vid_json)
        raw_vid_data = json_load(vid_json)

        subject_boxes = label_dict[i]['boxes']
        subject_boxes = [xywh_to_xyxy(box) for box in subject_boxes]

        # top_longest_track_ids = get_top_longest_tracks(vid_data, top_k=5)
        top_longest_track_ids = get_top_nearest_tracks(i, vid_data, subject_boxes, top_k=5)
        score_dict = evaluate(subject_boxes, [vid_data.track_map[i] for i in top_longest_track_ids])
        total_score_dict[i] = score_dict
        # if len(top_longest_track_ids) > 1:
        #     ambiguous_track[i] = len(top_longest_track_ids)
        #     pass

        if visualize and (not osp.isfile(save_path)):
            visualize_subject(raw_vid_data, top_longest_track_ids, DATA_DIR, save_path, subject_boxes)
    
    if json:
        fail_path = osp.join(save_dir, 'evaluate.json')
        json_save(total_score_dict, fail_path)
    pass

if __name__ == '__main__':
    exp_id = 'deepsort_v4-1'
    sub_id = 'v1'
    vid_save_dir = osp.join(SAVE_DIR, exp_id, f'video_sub-{sub_id}')
    os.makedirs(vid_save_dir, exist_ok=True)
    print(f'Find subject for {exp_id}')
    check_track_subject(
        save_dir=vid_save_dir, json_dir=osp.join(SAVE_DIR, exp_id, 'json'),
        visualize=False, json=True
    )
    pass

