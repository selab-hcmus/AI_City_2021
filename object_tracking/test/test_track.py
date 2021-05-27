import os, sys, pickle, json
import os.path as osp 
import pandas as pd 
from tqdm import tqdm
import numpy as np

from dataset.data_manager import (
    test_track_map, train_track_map, DATA_DIR
)
from test_utils import (
    json_save, json_load, xyxy_to_xywh
)
from object_tracking.tools import visualize
from object_tracking.deep_sort.iou_matching import iou

TRAIN_TRACKING_RESULT = './results/annotate_time_train'
TEST_TRACKING_RESULT = './results/annotate_time_test'
VIDEO_DIR = './results/video_smooth'
ID_TO_CHECK = [str(i) for i in range(10)]
IOU_ACCEPT_THRES = 0.2

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

def get_wrong_boxes(track_data):
    list_frames, list_boxes = track_data['frame_order'], track_data['boxes']
    prev_frame, prev_box = list_frames[0], xyxy_to_xywh(list_boxes[0])
    fail_frames = []
    for (cur_frame, cur_box) in zip(list_frames[1:], list_boxes[1:]):
        cur_box = xyxy_to_xywh(cur_box)
        iou_score = iou(prev_box, np.array([cur_box]))[0]
        
        if iou_score < IOU_ACCEPT_THRES or ():
            fail_frames.append(cur_frame)    
    pass

    return fail_frames

def main():
    df_dict = {
        'vid_ids': [], 'miss_frame': []
    }
    for i in tqdm(ID_TO_CHECK):
        vid_json = osp.join(TRAIN_TRACKING_RESULT, f'{i}.json')
        vid_data = json_load(vid_json)
        vid_path = osp.join(VIDEO_DIR, f'{i}.avi')

        fail_tracks = get_miss_frame_tracks(vid_data)
        df_dict['vid_ids'].append(i)
        df_dict['miss_frame'].append(fail_tracks)

        if not osp.isfile(vid_path):
            track_to_vids = [i for i in vid_data['track_map'].keys() if i not in fail_tracks]
            visualize(vid_data, track_to_vids, DATA_DIR, vid_path)

    df_stat = pd.DataFrame(df_dict)
    df_stat.to_csv('results/train_stat.csv', index=False)
    pass 

if __name__ == '__main__':
    main()
