"""Find subject track in list of tracking results
"""
import os, sys, pickle, json
import os.path as osp 
from tqdm import tqdm

from utils.data_manager import (
    test_track_map, train_track_map, DATA_DIR,
    TRAIN_TRACK_ORDER_JSON, TEST_TRACK_JSON,
)
from utils import AverageMeter, xywh_to_xyxy, json_load

from object_tracking.test.test_utils import calculate_iou, calculate_distance, SAVE_DIR
from object_tracking.test.evaluate_subject import evaluate
from object_tracking.library import VideoResult
from object_tracking.utils import subject_config

# GLOBAL VARIABLES
IOU_ACCEPT_THRES = subject_config['IOU_ACCEPT_THRES'] #not use yet
SCORE_THRES = subject_config['SCORE_THRES']
IOU_AVG_THRES = subject_config['IOU_AVG_THRES']

# test_track_order = {}
# for k, v in test_track_map.items():
#     test_track_order[str(v)] = k


## FUNCTIONS
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

def get_top_nearest_tracks(vid_data: VideoResult, label_boxes: list, top_k: int=5):  
    list_lens = []
    track_map = vid_data.track_map
    COMPARE_RANGE = subject_config['COMPARE_RANGE']

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

# label_dict = json_load(TRAIN_TRACK_ORDER_JSON) # Use to load groundtruth boxes
# label_dict = json_load(TEST_TRACK_ORDER_JSON) # Use to load groundtruth boxes
def find_subject_track(vid_data, subject_boxes: list):
    top_longest_track_ids = get_top_nearest_tracks(vid_data, subject_boxes, top_k=5)
    score_dict = evaluate(subject_boxes, [vid_data.track_map[i] for i in top_longest_track_ids])

    best_tracks = []
    for track_res in score_dict:
        if ((track_res['score'] > SCORE_THRES) and (track_res['iou_avg'] > IOU_AVG_THRES)) or \
                (track_res['score'] == 1.0):
            best_tracks.append((track_res['track_id'], track_res['iou_avg']))
        pass
    
    if len(best_tracks) == 0:
        return None, score_dict
        
    best_tracks = sorted(best_tracks, key = lambda val: val[1], reverse=True) # Sort with iou_avg
    subject_track_id = best_tracks[0][0]
    
    return subject_track_id, score_dict

def main2():
    test_track = json_load(TEST_TRACK_JSON)
    exp_id = 'test_deepsort_v4-3'
    track_dir = osp.join(SAVE_DIR, exp_id, 'json_full')
    miss_frame_ids = ['417', '168']
    cur_fail = ['417', '244', '320']
    
    test_ids = ["417","331","195","407","25","166","244","256","320"]
    result = {}
    for vid_id in tqdm(test_ids):
        vid_json_path = osp.join(track_dir, f'{vid_id}.json')
        vid_data = VideoResult(vid_json_path)
        
        subject_boxes = test_track[test_track_map[vid_id]]['boxes']
        subject_boxes = [xywh_to_xyxy(box) for box in subject_boxes]

        # vid_data = find_subject_track(vid_data, subject_boxes)
        subject_track_id, score_dict = find_subject_track(vid_data, subject_boxes)
        print(score_dict)
        result[vid_id] = subject_track_id
        pass
    
    print(result)
    pass
