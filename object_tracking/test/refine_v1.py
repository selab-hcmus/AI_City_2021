#TODO
"""Associate tracks
reference: https://github.com/feiaxyt/Winner_ECCV20_TAO/tree/main/tao_tracking_release/tao_post_processing

Requirements:
- tracking results must store feature --> pkl files

Input:
- Use the current format, include 'feature' to each track info in track_map 

Output:
- 
"""
import numpy as np
import os 
import os.path as osp
from tqdm import tqdm

from object_tracking.library.track_result import TrackResult
from object_tracking.library import VideoResult

SIMILARITY_THRES = 0.4
FRAME_DISTANCE_THRES = 2

def is_overlap(track_a: TrackResult, track_b: TrackResult):
    s1 = track_a.frame_order[0]
    e1 = track_a.frame_order[-1]
    s2 = track_b.frame_order[0]
    e2 = track_b.frame_order[-1]

    if (s2 - e1 > 0) or (s1 - e2 > 0):
        return False     

    return True

def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b.T)


def reid_similarity_v1(track_a: TrackResult, track_b: TrackResult):
    feat_a = np.mean(track_a.features)
    feat_b = np.mean(track_b.features)
    return cosine_similarity(feat_a, feat_b)

def reid_similarity_v2(track_a: TrackResult, track_b: TrackResult):
    return cosine_similarity(track_a.features[-1], track_b.features[0])

def get_frame_distance(track_a: TrackResult, track_b: TrackResult):
    return track_b.frame_order[0] - track_a.frame_order[-1]

def associate_track(list_tracks: list, similarity_type: str='v1'):
    N = len(list_tracks)
    assoc_res = []
    
    for i in range(0, N-1):
        track_a = list_tracks[i]
        if track_a.track_id != '1':
            continue
        for j in range(i+1, N):
            track_b = list_tracks[j]
            if track_b.track_id != '26':
                continue

            frame_distance = get_frame_distance(track_a, track_b)
            print(f'frame_distance: {frame_distance}')
            print(f'is_overlap: {is_overlap(track_a, track_b)}')
            if is_overlap(track_a, track_b) or frame_distance > FRAME_DISTANCE_THRES:
                continue
            
            # if no overlap, track b appears after track a
            score = None
            if similarity_type == 'v1':
                score = reid_similarity_v1(track_a, track_b)
            if similarity_type == 'v2':
                score = reid_similarity_v2(track_a, track_b)

            if score is None:
                print(f'Invalid similarity type')
                return

            print(f'score: {score}')
            if score > SIMILARITY_THRES:
                assoc_res.append((track_a.track_id, track_b.track_id, score))
            
    return assoc_res

def main():
    res_save_dir = '/home/ntphat/projects/AI_City_2021/object_tracking/results_exp/deepsort_v4/json'
    
    for file_name in tqdm(os.listdir(res_save_dir)):
        if file_name not in ['349.pkl']:
            continue
        
        print(f'Run {file_name}')
        file_path = osp.join(res_save_dir, file_name)
        vid_result = VideoResult(file_path)
        list_tracks = vid_result.get_list_tracks_by_time()

        assoc_res = associate_track(list_tracks)
        print(assoc_res)

        # example = list_tracks[0]
        # print(type(example.features))
        # print(len(example.features))
        # print(type(example.features[0]))
        # print(example.features[0].shape)
        # print(len(example.boxes))
        # print(type(example.feature))
        
    pass

if __name__ == '__main__':
    main()
