import json
import pickle 
import os 
import os.path as osp 
import numpy as np 
import cv2
from glob import glob
from tqdm import tqdm
import math
from detector import StopDetector, stop_detector
from relation_graph.utils import (
    visualize, PositionState, FollowState, Counter, length_vector,
    xyxy_to_xywh, smooth_distance, minus_vector, calculate_velocity_vector
)
import pandas as pd

from utils import prepare_dir
from utils.data_manager import DATA_DIR, RESULT_DIR
from object_tracking.tools.visualize import visualize
from relation_graph.config import (
    VELOCITY_SKIP_FRAME,
    # VELOCITY_COSINE_THRES,
    # DISTANCE_COSINE_THRES,
    # POSITION_COSINE_THRES,
    THRES_LEVEL,
    DISTANCE_THRES,
    MAX_TRAJ_THRES_LEVEL,
    FOLLOW_STATE_THRES
)
from statistics import mean
RELATION_SAVE_DIR = osp.join(RESULT_DIR, 'relation_graph')
# TRAIN_TRACKING_RESULT = '../../object_tracking/results_exp/test_deepsort_v4-1/json_stop'
TEST_TRACKING_RESULT = osp.join(RESULT_DIR, 'object_tracking_exp', 'test_deepsort_v4-3/json_stop') 

EXP_ID = 'relation_June14_13h14pm_reducecountthres'
SAVE_DIR = prepare_dir(osp.join(RELATION_SAVE_DIR, EXP_ID))
SAVE_DIR_CSV = SAVE_DIR
SAVE_DIR_VIDEO = prepare_dir(osp.join(SAVE_DIR, "visualize")) 
# DATA_DIR = "../../dataset"
def get_potential_track(track_dir: str):
    ans = []
    for json_path in tqdm(glob(track_dir +'/*.json')):
        json_data = json.load(open(json_path, 'r'))
        if json_data['n_frames'] >= 5:
            vid_id = json_path.split('/')[-1].split('.')[0]
            # print(f'Work on id: {vid_id}')
            ans.append((vid_id, json_data))
            # return ans
    return ans
    
def calculate_distance_vector(coor_a, coor_b, skip_frame=2):
    if skip_frame > len(coor_a):
        skip_frame = 1
    # print(len(coor_a), len(coor_b), skip_frame)
    dis_list = [minus_vector(coor_a[i], coor_b[i]) for i in range(len(coor_a) - skip_frame)]
    return dis_list

def cosine_similarity(vect_a, vect_b):
    if isinstance(vect_a, list):
        vect_a = np.array(vect_a)
    if isinstance(vect_b, list):
        vect_b = np.array(vect_b)

    return np.dot(vect_a, vect_b)/(np.linalg.norm(vect_a)*np.linalg.norm(vect_b)) #default: L2 norm

def get_interval_to_check(track_frame_order, overlapped_frame_order):
    # start_idx, end_idx = 0, 0
    # for i, frame_order in enumerate(track_frame_order):
    #     if frame_order == overlapped_frame_order[0]:
    #         start_idx = i 
    #     if frame_order == overlapped_frame_order[-1]:
    #         end_idx = i 
    #         break 
    return [index for index, value in enumerate(track_frame_order) if value in overlapped_frame_order]

    # return start_idx, end_idx

def get_follow_relation(track_a: dict, track_b: dict, pos_thres=0, v_thres=0, traj_thres=0):
    list_frame_a = track_a['frame_order']
    list_frame_b = track_b['frame_order']
    overlapped_frames = list(set(list_frame_a).intersection(set(list_frame_b)))
    overlapped_frames.sort() 
    n = len(overlapped_frames)
    if n <= 4:
        return FollowState.NO_RELATION, None, None # Two tracks are not overlapped

    # start_idx_a, end_idx_a = get_interval_to_check(list_frame_a, overlapped_frames)
    # start_idx_b, end_idx_b = get_interval_to_check(list_frame_b, overlapped_frames)
    
    # coor_a = [xyxy_to_xywh(box) for box in track_a['boxes'][start_idx_a:end_idx_a+1]]
    # coor_b = [xyxy_to_xywh(box) for box in track_b['boxes'][start_idx_b:end_idx_b+1]]

    # get all each track's indices representing the values in overlapped_frames
    idx_a = get_interval_to_check(list_frame_a, overlapped_frames)
    idx_b = get_interval_to_check(list_frame_b, overlapped_frames)

    coor_a = [xyxy_to_xywh(box) for box in [track_a['boxes'][i] for i in idx_a]]
    coor_b = [xyxy_to_xywh(box) for box in [track_b['boxes'][i] for i in idx_b]]

    # print(coor_a)
    # print(coor_b)
    
    # print(f'n: {n}')
    # print(f'len(coor_a): {len(coor_a)}')
    # print(f'len(coor_b): {len(coor_b)}')

    velocity_vect_a = calculate_velocity_vector(coor_a, skip_frame=VELOCITY_SKIP_FRAME)
    velocity_vect_b = calculate_velocity_vector(coor_b, skip_frame=VELOCITY_SKIP_FRAME)
    distance_vect_ab = calculate_distance_vector(coor_a, coor_b, skip_frame=VELOCITY_SKIP_FRAME)
    
    avg_distance = mean([length_vector(v) for v in distance_vect_ab])

    # print(f'len(velocity_vect_a): {len(velocity_vect_a)}')
    # print(f'len(velocity_vect_b): {len(velocity_vect_b)}')
    # print(f'len(distance_vect_ab): {len(distance_vect_ab)}')

    n = len(distance_vect_ab)
    cosine_va_ab = [cosine_similarity(velocity_vect_a[i], distance_vect_ab[i]) for i in range(n)]
    cosine_vb_ab = [cosine_similarity(velocity_vect_b[i], distance_vect_ab[i]) for i in range(n)]
    cosine_va_vb = [cosine_similarity(velocity_vect_a[i], velocity_vect_b[i]) for i in range(n)]

    avg_cos = mean(cosine_va_ab)

    # print("coor_a \n", coor_a)
    # print("coor_b \n", coor_b)
    # print("velocity_vect_a: \n", velocity_vect_a)
    # print("velocity_vect_b: \n", velocity_vect_b)
    # print("distance_vect_ab: \n", distance_vect_ab)
    # print("cosine_va_ab: \n", cosine_va_ab)
    # print("cosine_vb_ab: \n", cosine_vb_ab)
    # print("cosine_va_vb: \n", cosine_va_vb)

    # check position (A behind B or B behind A)
    position_state = []
    for i in range(n):
        x, y = velocity_vect_a[i]
        if x**2 + y**2 <= DISTANCE_THRES:
            position_state.append(PositionState.NO_RELATION)
            continue

        if cosine_va_ab[i] >= pos_thres:
            position_state.append(PositionState.A_BEHIND_B)
        elif cosine_va_ab[i] <= -pos_thres:
            position_state.append(PositionState.B_BEHIND_A)
        else:
            position_state.append(PositionState.NO_RELATION)
    
    # check relation (A folow B or B follow A or no relation)
    # follow_state = {
    #     FollowState.NO_RELATION: 0,
    #     FollowState.A_FOLLOW_B: 0,
    #     FollowState.B_FOLLOW_A: 0
    # }

    follow_state = []
    follow_state_counter = Counter()
    for i in range(n):
        if position_state[i] == PositionState.NO_RELATION:
            continue
        if position_state[i] == PositionState.A_BEHIND_B:
            if (not (cosine_va_vb[i] >= v_thres)) or abs(cosine_va_ab[i]) < traj_thres:
                follow_state.append(FollowState.NO_RELATION)
                follow_state_counter.update(FollowState.NO_RELATION)
            else:
                follow_state.append(FollowState.A_FOLLOW_B)
                follow_state_counter.update(FollowState.A_FOLLOW_B)
            pass 

        if position_state[i] == PositionState.B_BEHIND_A:
            # if abs(cosine_vb_ab[i]) < FOLLOW_COSINE_THRES:
            if (not (cosine_va_vb[i] >= v_thres)) or abs(cosine_va_ab[i]) < traj_thres:
                follow_state.append(FollowState.NO_RELATION)
                follow_state_counter.update(FollowState.NO_RELATION)                
            else:
                follow_state.append(FollowState.B_FOLLOW_A)
                follow_state_counter.update(FollowState.B_FOLLOW_A)
            pass
        pass
    
    if follow_state_counter.find_longest_state(FollowState.B_FOLLOW_A) >= FOLLOW_STATE_THRES and follow_state_counter.find_longest_state(FollowState.A_FOLLOW_B) >= FOLLOW_STATE_THRES:
        ans = FollowState.NO_RELATION
    else:
        ans = follow_state_counter.get_famous_value()
    return ans, avg_distance, avg_cos

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

def find_all_follow_relation(val_id, data):
    sub_key = data.get("subject", None)
    if sub_key is None:
        return None
    track_map = data['track_map']
    list_track_ids = list(track_map.keys())
    ans = {
        "key": val_id,
        "subject": [sub_key],
        "follow": [],
        "follow_by": [],
        "num_frames": data["n_frames"]
    }
    print(val_id, sub_key)
    ans_follow = []
    ans_follow_by = []
    for max_thres_traj in MAX_TRAJ_THRES_LEVEL:
        for thres in THRES_LEVEL:
            if len(ans["follow"]) and len(ans["follow_by"]):
                break
            sub_ans_follow = []
            sub_ans_follow_by = []
            for key in list_track_ids:
                if key == sub_key:
                    continue
                if key in data["stop_tracks"]:
                    continue
                isFollow, avg_distance, avg_cos = get_follow_relation(track_map[sub_key], track_map[key], pos_thres=math.cos(math.pi/15), v_thres=thres, traj_thres=max_thres_traj)
                # print(key, isFollow)
                if isFollow == FollowState.A_FOLLOW_B and not ans_follow:
                    sub_ans_follow.append({
                        "key": key,
                        "avg_dist": avg_distance,
                        "avg_cos": avg_cos
                        })
    
                elif isFollow == FollowState.B_FOLLOW_A and not ans_follow_by:
                    sub_ans_follow_by.append({
                        "key": key,
                        "avg_dist": avg_distance,
                        "avg_cos": avg_cos
                        })
                # input()
                pass
            if sub_ans_follow:
                ans_follow.extend(sub_ans_follow)
            if sub_ans_follow_by:
                ans_follow_by.extend(sub_ans_follow_by)
    
    if len(ans_follow) >= 1:
        min_dist_key = min(ans_follow, key=lambda d: d['avg_dist'])["key"]
        max_cos_key = max(ans_follow, key=lambda d: d['avg_cos'])["key"]
        ans["follow"].extend(list(set([min_dist_key, max_cos_key])))
    
    if len(ans_follow_by) >= 1:
        min_dist_key = min(ans_follow_by, key=lambda d: d['avg_dist'])["key"]
        min_cos_key = min(ans_follow_by, key=lambda d: d['avg_cos'])["key"]
        ans["follow_by"].extend(list(set([min_dist_key, min_cos_key])))

    print("follow: ", ans["follow"])
    print("follow_by: ", ans["follow_by"])
    return ans


def main():
    stop_detector = StopDetector() 
    json_data = get_potential_track(TEST_TRACKING_RESULT)
    count = 0
    ans = []
    for val_id, data in tqdm(json_data):
        row_dict = find_all_follow_relation(val_id, data)
        if row_dict is None:
            continue
        visualize(data, None, DATA_DIR, SAVE_DIR_VIDEO, row_dict, val_id)
        ans.append(row_dict)
        count += 1
        # if (count >= 10):
        #     break

    ans_df = pd.DataFrame(data=ans)
    ans_df.to_csv(osp.join(SAVE_DIR_CSV, "ans.csv"), index=False)

    # if json_data is None:
    #     print(f'No data available')
    #     return

    # list_frames = json_data['list_frames']
    # n_tracks = json_data['n_tracks']
    # track_map = json_data['track_map']
    # list_track_ids = list(track_map.keys())

    # relation_res = {}
    # longest_track_id = get_longest_tracklet(track_map)
    # print(f'vid_id: {vid_id}')

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
