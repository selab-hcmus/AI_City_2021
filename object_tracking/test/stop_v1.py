import os 
import os.path as osp
from tqdm import tqdm

from utils import json_load, json_save
from object_tracking.library import VideoResult, TrackResult
from object_tracking.test.test_utils import SAVE_DIR, calculate_iou, calculate_distance
track_res_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-2', 'json_subject')
save_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-2', 'json_stop')
print(f'Save result to {save_dir}')
os.makedirs(save_dir)

STOP_IOU_THRES = 0.6
def is_track_stop(track_data: TrackResult):
    first_box = track_data.boxes[0]
    last_box = track_data.boxes[-1]
    if calculate_iou(last_box, first_box) > STOP_IOU_THRES:
        return True
    return False

def find_stop_track(video_data: VideoResult):
    stop_tracks = []
    for track_id in video_data.track_map:
        if is_track_stop(video_data.track_map[track_id]):
            stop_tracks.append(track_id)
    
    return stop_tracks

def main():
    list_files = os.listdir(track_res_dir)
    # list_ids = [407]
    # list_files = [f'{i}.json' for i in list_ids]
    for fname in tqdm(list_files):
        inp = osp.join(track_res_dir, fname)
        vid_data = VideoResult(inp)
        stop_tracks = find_stop_track(vid_data)
        # print(f"subject: {vid_data.subject}")
        # print(f'{fname}: {stop_tracks} stop')

        save_path = osp.join(save_dir, fname)
        old_data = json_load(inp)
        old_data['stop_tracks'] = stop_tracks
        json_save(old_data, save_path)

        pass
    pass 


if __name__ == '__main__':
    main()
