import json
from object_tracking.deep_sort import track
import pickle 
import os 
import os.path as osp 
import numpy as np 
import cv2
from glob import glob
from tqdm import tqdm

from detector import StopDetector, stop_detector
from utils import (
    visualize, PositionState, FollowState, Counter, 
    xyxy_to_xywh, smooth_distance, minus_vector, calculate_velocity_vector
)

VELOCITY_SKIP_FRAME = 4
FOLLOW_COSINE_THRES = 0.8

TRAIN_TRACKING_RESULT = '../object_tracking/results/annotate_time_train'
TEST_TRACKING_RESULT = '../object_tracking/results/annotate_time_test'

SAVE_DIR = './data'
DATA_DIR = '../dataset'

def get_potential_track(track_dir: str):
    for json_path in glob(track_dir +'/*.json'):
        json_data = json.load(open(json_path, 'r'))
        if json_data['n_frames'] > 50:
            vid_id = json_path.split('/')[-1].split('.')[0]
            print(f'Work on id: {vid_id}')
            return json_data, vid_id
        pass
    pass
    return None, None
    
def calculate_distance_vector(coor_a, coor_b, skip_frame=2):
    if skip_frame > len(coor_a):
        skip_frame = 1
    dis_list = [minus_vector(coor_a[i], coor_b[i]) for i in range(len(coor_a) - skip_frame)]
    
    return dis_list

def cosine_similarity(vect_a, vect_b):
    if isinstance(vect_a, list):
        vect_a = np.array(vect_a)
    if isinstance(vect_b, list):
        vect_b = np.array(vect_b)

    return np.dot(vect_a, vect_b)/(np.linalg.norm(vect_a)*np.linalg.norm(vect_b)) #default: L2 norm

def get_interval_to_check(track_frame_order, overlapped_frame_order):
    start_idx, end_idx = 0, 0
    for i, frame_order in enumerate(track_frame_order):
        if frame_order == overlapped_frame_order[0]:
            start_idx = i 
        if frame_order == overlapped_frame_order[-1]:
            end_idx = i 
            break 

    return start_idx, end_idx

def get_follow_relation(track_a: dict, track_b: dict):
    list_frame_a = track_a['frame_order']
    list_frame_b = track_b['frame_order']
    overlapped_frames = list(set(list_frame_a).intersection(set(list_frame_b)))
    overlapped_frames.sort() 
    n = len(overlapped_frames)
    if n <= 4:
        return False # Two tracks are not overlapped

    start_idx_a, end_idx_a = get_interval_to_check(list_frame_a, overlapped_frames)
    start_idx_b, end_idx_b = get_interval_to_check(list_frame_b, overlapped_frames)
    
    coor_a = [xyxy_to_xywh(box) for box in track_a['boxes'][start_idx_a:end_idx_a+1]]
    coor_b = [xyxy_to_xywh(box) for box in track_b['boxes'][start_idx_b:end_idx_b+1]]
    
    velocity_vect_a = calculate_velocity_vector(coor_a, skip_frame=VELOCITY_SKIP_FRAME)
    velocity_vect_b = calculate_velocity_vector(coor_b, skip_frame=VELOCITY_SKIP_FRAME)
    distance_vect_ab = calculate_distance_vector(coor_a, coor_b, skip_frame=VELOCITY_SKIP_FRAME)
    
    print(f'n: {n}')
    print(f'len(coor_a): {len(coor_a)}')
    print(f'len(coor_b): {len(coor_b)}')
    print(f'len(velocity_vect_a): {len(velocity_vect_a)}')
    print(f'len(velocity_vect_b): {len(velocity_vect_b)}')
    print(f'len(distance_vect_ab): {len(distance_vect_ab)}')


    cosine_va_ab = [cosine_similarity(velocity_vect_a[i], distance_vect_ab[i]) for i in range(n)]
    cosine_vb_ab = [cosine_similarity(velocity_vect_b[i], distance_vect_ab[i]) for i in range(n)]
    cosine_va_vb = [cosine_similarity(velocity_vect_a[i], velocity_vect_b[i]) for i in range(n)]

    # check position (A behind B or B behind A)
    position_state = []
    for i in range(n):
        if cosine_va_ab[i] > 0:
            position_state.append(PositionState.A_BEHIND_B)
        else:
            position_state.append(PositionState.B_BEHIND_A)
        pass
    
    # check relation (A folow B or B follow A or no relation)
    follow_state = []
    follow_state_counter = Counter()
    for i in range(n):
        if position_state[i] == PositionState.A_BEHIND_B:
            if abs(cosine_va_ab[i]) < FOLLOW_COSINE_THRES:
                follow_state.append(FollowState.NO_RELATION)
                follow_state_counter.update(FollowState.NO_RELATION)
            else:
                follow_state.append(FollowState.A_FOLLOW_B)
                follow_state_counter.update(FollowState.A_FOLLOW_B)
            pass 

        if position_state[i] == PositionState.B_BEHIND_A:
            if abs(cosine_vb_ab[i]) < FOLLOW_COSINE_THRES:
                follow_state.append(FollowState.NO_RELATION)
                follow_state_counter.update(FollowState.NO_RELATION)                
            else:
                follow_state.append(FollowState.B_FOLLOW_A)
                follow_state_counter.update(FollowState.B_FOLLOW_A)
            pass
        pass

    return follow_state_counter.get_famous_value()

def get_longest_tracklet(track_map: dict):
    longest_track_id = None
    for track_id in track_map.keys():
        if longest_track_id is None:
            longest_track_id = track_id
        else:
            if len(track_map[track_id]['frame_order']) > len(track_map[longest_track_id]['frame_order']):
                longest_track_id = track_id
        pass

    return longest_track_id

def main():
    stop_detector = StopDetector()
    json_data, vid_id = get_potential_track(TRAIN_TRACKING_RESULT)
    if json_data is None:
        print(f'No data available')
        return

    list_frames = json_data['list_frames']
    n_tracks = json_data['n_tracks']
    track_map = json_data['track_map']
    list_track_ids = list(track_map.keys())

    relation_res = {}
    longest_track_id = get_longest_tracklet(track_map)
    print(f'vid_id: {vid_id}')

    # vid_save_path = f'./{vid_id}.avi'
    # visualize(json_data, ['1', '4'], DATA_DIR, vid_save_path)
    # print(f'save video to {vid_save_path}')

    # for track_id in ['4', '1']:
    #     if stop_detector.process(track_map[track_id]):
    #         print(f'Track {track_id} stopped')
    #     else:
    #         print(f'Track {track_id} did not stop')
    #     pass

    # print('Run relation checking')
    # for track_id in tqdm(list_track_ids):
    #     print(f'{longest_track_id}_{track_id}')
    #     relation_state = get_follow_relation(track_map[longest_track_id], track_map[track_id])
    #     relation_name = FollowState.RELATION_NAME[relation_state]
    #     relation_res[f'{longest_track_id}-{track_id}'] = relation_name
    #     pass

    # with open(f'./relation_result.json', 'w') as f:
    #     json.dump(relation_res, f)
    # pass 

    # vid_save_path = f'./{vid_id}.avi'
    # visualize(json_data, list_track_ids, DATA_DIR, vid_save_path)
    # print(f'save video to {vid_save_path}')

if __name__ == '__main__':
    main()
