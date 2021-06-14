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

from utils.data_manager import (
    test_track_map, train_track_map, DATA_DIR,
    TRAIN_TRACK_ORDER_JSON, TEST_TRACK_JSON,
)
from utils import (
    AverageMeter, xyxy_to_xywh, xywh_to_xyxy, json_save, json_load, pickle_save, pickle_load
)
from object_tracking.test.test_utils import (
    a_substract_b, is_miss_frame, get_miss_frame_tracks, calculate_iou, calculate_distance,
    SAVE_DIR
)
from object_tracking.test.evaluate_subject import evaluate, eda_score_dict #, TRAIN_SVO_IDS
from object_tracking.tools import visualize, visualize_subject
from object_tracking.library import VideoResult


# ID_TO_CHECK = [str(i) for i in [5, 6, 9, 84]]
IOU_ACCEPT_THRES = 0.2 #not use yet

test_track_order = {}
for k, v in test_track_map.items():
    test_track_order[str(v)] = k

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

# label_dict = json_load(TRAIN_TRACK_ORDER_JSON) # Use to load groundtruth boxes
# label_dict = json_load(TEST_TRACK_ORDER_JSON) # Use to load groundtruth boxes
test_track = json_load(TEST_TRACK_JSON)
def check_track_subject(save_dir: str, vid_save_dir: str, file_dir: str,
                        visualize=False, json=True, file_mode: str='json'):

    list_keys = []
    for fname in os.listdir(file_dir):
        key = fname.split('.')[0]
        list_keys.append(key)
    
    list_keys = list(set(list_keys))
    if len(list_keys) == 0:
        print(f'Found no tracking result in {file_dir}')
        return

    print(f'Start finding main subject track')
    total_score_dict = {}
    for i in tqdm(list_keys):
        save_path = osp.join(vid_save_dir, f'{i}.avi')
        vid_json = osp.join(file_dir, f'{i}.{file_mode}')
        vid_data = VideoResult(vid_json)

        subject_boxes = test_track[test_track_order[i]]['boxes']
        subject_boxes = [xywh_to_xyxy(box) for box in subject_boxes]

        # top_longest_track_ids = get_top_longest_tracks(vid_data, top_k=5)
        top_longest_track_ids = get_top_nearest_tracks(i, vid_data, subject_boxes, top_k=5)
        score_dict = evaluate(subject_boxes, [vid_data.track_map[i] for i in top_longest_track_ids])
        total_score_dict[i] = score_dict

    if json:
        fail_path = osp.join(save_dir, 'all_score_dict.json')
        json_save(total_score_dict, fail_path)
    
    return total_score_dict

def main():
    exp_id = 'test_deepsort_v4-3'
    sub_id = 'subject_v1'
    mode = 'pkl'
    VISUALIZE=False

    track_dir = osp.join(SAVE_DIR, exp_id)
    save_dir = osp.join(track_dir, sub_id)
    vid_save_dir = osp.join(save_dir, 'video')
    vid_fail_dir = osp.join(save_dir, 'video_fail')
    
    os.makedirs(vid_save_dir, exist_ok=True)
    os.makedirs(vid_fail_dir, exist_ok=True)

    print(f'Find subject for {exp_id}, save result to {save_dir}')
    total_score_dict = check_track_subject(
        save_dir=save_dir, vid_save_dir=vid_save_dir, file_dir=osp.join(track_dir, mode),
        visualize=False, json=True, file_mode=mode
    )
    df_score, list_csv = eda_score_dict(total_score_dict)
    
    # Save subject info to file
    old_json_dir = osp.join(track_dir, mode)
    new_json_dir = osp.join(track_dir, f'{mode}_subject')
    print(f'{mode} with subject is stored in {new_json_dir}')
    os.makedirs(new_json_dir, exist_ok=True)
    result_ids = []
    for sample in tqdm(list_csv):
        json_path = osp.join(old_json_dir, f"{sample['track_id']}.{mode}")
        vid_data = VideoResult(json_path)
        if sample['is_perfect'] == True:    
            best_track_id = sample['best_tracks'][0]
            vid_data.set_subject(best_track_id)
        
        json_path = osp.join(new_json_dir, f"{sample['track_id']}.{mode}")
        if osp.isfile(json_path):
            continue
        else:
            result_ids.append(sample['track_id'])
        vid_data.to_json(json_path, is_feat=(mode == 'pkl'))
        if VISUALIZE:
            vid_save_path = osp.join(vid_save_dir, f"{sample['track_id']}.avi")
            vid_data.visualize(vid_save_path)
    
    print(f'Print ids: {result_ids}')
    df_score.to_csv(osp.join(save_dir, 'eval_score.csv'), index=False)
    
    if VISUALIZE:
        fail_ids = df_score[df_score['is_perfect'] == False]['track_id'].values.tolist()
        fail_ids.sort()
        for i in fail_ids:
            shutil.copyfile(
                src = osp.join(vid_save_dir, f'{i}.avi'),
                dst = osp.join(vid_fail_dir, f'{i}.avi')
            )
    pass

def main_1():
    print('Convert pkl to json')
    exp_id = 'test_deepsort_v4-3'
    old_dir = osp.join(SAVE_DIR, exp_id, 'pkl')
    new_dir = osp.join(SAVE_DIR, exp_id, 'json')
    os.makedirs(new_dir, exist_ok=True)

    for fname in tqdm(os.listdir(old_dir)):
        pickle_path = osp.join(old_dir, fname)
        vid_data = VideoResult(pickle_path)
        json_save_path = osp.join(new_dir, fname.replace('.pkl', '.json'))
        vid_data.to_json(save_path=json_save_path, is_feat=False)

        pass

if __name__ == '__main__':
    # main_1()
    main()
    