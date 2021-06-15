import os 
import os.path as osp
from tqdm import tqdm

from utils import AverageMeter
from object_tracking.library import VideoResult, TrackResult
from object_tracking.test.test_utils import SAVE_DIR, calculate_iou
from object_tracking.utils import stop_config

STOP_IOU_THRES = stop_config['STOP_IOU_THRES']
COMPARE_RANGE = stop_config['COMPARE_RANGE']

def is_track_stop(track_data: TrackResult, k=COMPARE_RANGE):
    meter = AverageMeter()
    n = min(k, len(track_data.boxes)//2)
    first_boxes = track_data.boxes[:n]
    last_boxes = track_data.boxes[-n:]
    for i in range(n):
        first_box = first_boxes[i]
        last_box = last_boxes[n-1-i]
        meter.update(calculate_iou(last_box, first_box))
    
    if meter.avg > STOP_IOU_THRES:
        return True
    return False

def find_stop_track(video_data: VideoResult):
    stop_tracks = []
    for track_id in video_data.track_map:
        if is_track_stop(video_data.track_map[track_id]):
            stop_tracks.append(track_id)
    
    return stop_tracks

def main():
    VISUALIZE=False
    mode='pkl'

    track_res_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-3', f'{mode}_subject')
    save_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-3', f'{mode}_stop')
    vid_save_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-3', 'video_stop')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vid_save_dir, exist_ok=True)

    print(f'Save files to {save_dir}')
    list_files = os.listdir(track_res_dir)
    # list_files = [f'{i}.{mode}' for i in list_ids]
    for fname in tqdm(list_files):
        inp = osp.join(track_res_dir, fname)
        track_id = fname.split('.')[0]
        save_path = osp.join(save_dir, fname)
        # if osp.isfile(save_path):
        #     continue
        
        vid_data = VideoResult(inp)
        stop_tracks = find_stop_track(vid_data)
        vid_data.set_stop_tracks(stop_tracks)
        vid_data.to_json(save_path, is_feat=(mode=='pkl'))
        
        if VISUALIZE:
            vid_save_path = osp.join(vid_save_dir, f'{track_id}.avi')
            vid_data.visualize(vid_save_path)
        pass
    pass 


# if __name__ == '__main__':
#     main()
