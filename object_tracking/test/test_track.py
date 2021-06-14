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
    TRAIN_TRACK_ORDER_JSON, TEST_TRACK_ORDER_JSON
)
from utils import (
    AverageMeter, xyxy_to_xywh, xywh_to_xyxy
)
from test_utils import (
    json_save, json_load, a_substract_b,
    is_miss_frame, get_miss_frame_tracks
)
from object_tracking.tools import visualize, visualize_subject
from object_tracking.deep_sort.iou_matching import iou

TRAIN_TRACKING_RESULT = './results/annotate_time_train'
TEST_TRACKING_RESULT = './results/annotate_time_test'
VIDEO_DIR = './results/video_smooth'
ID_TO_CHECK = [str(i) for i in [5, 6, 9, 84]]
IOU_ACCEPT_THRES = 0.2

## Get misaligned tracks
def distance(box_a: list, box_b: list):
    # x, y, w, h
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    xa += wa/2
    ya += ha/2
    xb += wb/2
    yb += hb/2
    return np.sqrt((xa-xb)**2 + (ya-yb)**2)

def mis_align_box(cur_distance: float, avg_distance: float):
    if cur_distance > 2*avg_distance:
        return True
    return False

def get_wrong_frames(track_data):
    list_frames, list_boxes = track_data['frame_order'], track_data['boxes']
    prev_frame, prev_box = list_frames[0], xyxy_to_xywh(list_boxes[0])
    fail_frames = []
    distance_meter = AverageMeter()
    for i, (cur_frame, cur_box) in enumerate(zip(list_frames[1:], list_boxes[1:])):
        cur_box = xyxy_to_xywh(cur_box)
        iou_score = iou(np.array(prev_box), np.array([cur_box]))[0]
        cur_distance = distance(cur_box, prev_box)

        if iou_score < IOU_ACCEPT_THRES or mis_align_box(cur_distance, distance_meter.avg):
            fail_frames.append((i, cur_frame))
            #TODO: do sth to fix this frame
        else:
            distance_meter.update(cur_distance)
        
        prev_frame, prev_box = cur_frame, cur_box

    return fail_frames

def get_wrong_tracks(vid_data: dict, list_track_ids: list = None):
    fail_info = []
    
    if list_track_ids is None:
        list_track_ids = list(vid_data['track_map'].keys())
    for track_id in list_track_ids:
        track_data = vid_data['track_map'][track_id]
        missed_frames = get_wrong_frames(track_data)
        if len(missed_frames) > 0:
            fail_info.append({'track_id': track_id, 'fail_frames': missed_frames})
        pass
    fail_track_ids = list(fail_info.keys())
    return fail_track_ids, fail_info

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

def get_top_nearest_tracks(vid_data: dict, label_boxes: list, top_k: int=5):  
    iou_thres = 0.5
    list_lens = []
    track_map = vid_data['track_map']
    COMPARE_RANGE = 10
    frame_idx_to_compare = 0

    for track_id in track_map:
        track_order = track_map[track_id]['frame_order']

        iou_score = AverageMeter()
        track_range = min(len(track_order), COMPARE_RANGE)
        for frame_idx_to_compare in range(track_range):
            first_frame_idx = track_order[frame_idx_to_compare]
            track_first_box = track_map[track_id]['boxes'][frame_idx_to_compare]

            # track_first_box = xyxy_to_xywh(track_first_box)
            cor_subject_box = label_boxes[first_frame_idx]
            # cor_subject_box = xyxy_to_xywh(cor_subject_box)
            
            # iou_score = iou(cor_subject_box, np.array([track_first_box]))
            cur_score = calculate_iou(track_first_box, cor_subject_box)
            iou_score.update(cur_score)

        if iou_score.avg >= iou_thres:
            list_lens.append( (track_id, len(track_order)) )
        
    sorted(list_lens, key=lambda val: val[1])

    if top_k > len(list_lens):
        top_k = len(list_lens)
    
    return [i[0] for i in list_lens[:top_k]]



label_dict = json_load(TRAIN_TRACK_ORDER_JSON)
def check_track_subject(list_keys: list=None, visualize=False, exp_id: str='v1', json=True):
    save_dir = osp.join('results_compare', f'subject_{exp_id}')
    os.makedirs(save_dir, exist_ok=True)

    ambiguous_track = {}

    if list_keys is None:
        list_keys = list(label_dict.keys())
    # label_dict = json_load(TRAIN_TRACK_ORDER_JSON)    
    for i in tqdm(list_keys):
        save_path = osp.join(save_dir, f'{i}.avi')
        
        vid_json = osp.join(TRAIN_TRACKING_RESULT, f'{i}.json')
        vid_data = json_load(vid_json)

        subject_boxes = label_dict[i]['boxes']
        subject_boxes = [xywh_to_xyxy(box) for box in subject_boxes]

        # top_longest_track_ids = get_top_longest_tracks(vid_data, top_k=5)
        top_longest_track_ids = get_top_nearest_tracks(vid_data, subject_boxes, top_k=5)
        if len(top_longest_track_ids) > 1:
            # print(f'{i}: {len(top_longest_track_ids)}')
            ambiguous_track[i] = len(top_longest_track_ids)
            pass

        if visualize and (not osp.isfile(save_path)):
            visualize_subject(vid_data, top_longest_track_ids, DATA_DIR, save_path, subject_boxes)
        # print(f'save result to {save_path}')
    
    if json:
        with open(f'results_compare/not_sure_{exp_id}.json', 'w') as f:
            json.dump(ambiguous_track, f, indent=2)    
    pass

def main():
    df_dict = {
        'vid_ids': [], 'miss_frame': []
    }
    for i in tqdm(ID_TO_CHECK):
        vid_json = osp.join(TRAIN_TRACKING_RESULT, f'{i}.json')
        vid_data = json_load(vid_json)
        vid_path = osp.join(VIDEO_DIR, f'{i}.avi')

        fail_tracks = get_miss_frame_tracks(vid_data)
        right_tracks_v1 = [i for i in vid_data['track_map'].keys() if i not in fail_tracks]
        
        fail_tracks, fail_info = get_wrong_tracks(vid_data, right_tracks_v1)
        right_tracks_v2 = [i for i in right_tracks_v1 if i not in fail_tracks]

        df_dict['vid_ids'].append(i)
        df_dict['miss_frame'].append(fail_tracks)

        if not osp.isfile(vid_path):
            track_to_vids = right_tracks_v1
            visualize(vid_data, track_to_vids, DATA_DIR, vid_path)

    df_stat = pd.DataFrame(df_dict)
    df_stat.to_csv('results/train_stat.csv', index=False)
    pass 

def main2():
    not_sure_v1 = 'results_compare/not_sure_v1.json'
    not_sure_v1 = json_load(not_sure_v1)
    list_v1_fail = list(not_sure_v1.keys())
    print(f'not_sure_v1: {len(list_v1_fail)}')
    
    
    not_sure_v2 = 'results_compare/not_sure_v2.json'
    not_sure_v2 = json_load(not_sure_v2)
    list_v2_fail = list(not_sure_v2.keys())
    print(f'not_sure_v2: {len(list_v2_fail)}')

    v1_not_v2 = a_substract_b(list_v1_fail, list_v2_fail)
    v2_not_v1 = a_substract_b(list_v2_fail, list_v1_fail)
    print(f'v1_not_v2 : {len(v1_not_v2)}')
    print(f'v2_not_v1 : {len(v2_not_v1)}')
    print(v2_not_v1)
    # main()

    tmp_dir = 'results_compare/tmp'
    if osp.isdir(tmp_dir):
        os.removedirs(tmp_dir)
    else:
        os.makedirs(tmp_dir, exist_ok=True)
    for fname in v2_not_v1:
        shutil.copyfile(
            src = osp.join('results_compare/subject', f'{fname}.avi'), 
            dst = osp.join(tmp_dir, f'{fname}.avi')
        )
    print(f'copy files to {tmp_dir}')
    pass


from object_tracking.deep_sort.postprocessing import remove_track_with_extreme_box
def main3():
    list_ids = list(label_dict.keys()) # ['1638']
    save_dir = 'results_exp/Exp_v2'
    os.makedirs(save_dir, exist_ok=True)
    
    logger = logging.getLogger('Aic')
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler('results_exp/Exp_v2.log')
    f_format = logging.Formatter('%(levelname)s: %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    for i in tqdm(list_ids):
        vid_json = osp.join(TRAIN_TRACKING_RESULT, f'{i}.json')
        vid_data = json_load(vid_json)
        new_vid_data, fail_ids = remove_track_with_extreme_box(vid_data)
        logger.info(f'{i}, {len(fail_ids)} fail ids: {fail_ids}')
        
        subject_boxes = label_dict[i]['boxes']
        subject_boxes = [xywh_to_xyxy(box) for box in subject_boxes]

        visualize_subject(vid_data, fail_ids, DATA_DIR, osp.join(save_dir, f'{i}.avi'), subject_boxes)
    pass

if __name__ == '__main__':
    main3()
    # check_track_subject(
    #     list_keys=['1638'],
    #     visualize=True,
    #     exp_id='v2',
    #     json=False
    # )
    pass
