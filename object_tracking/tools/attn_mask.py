import json 
import os 
import os.path as osp
import cv2 
import numpy as np
from tqdm import tqdm

from utils import (
    json_load, json_save, pickle_load, pickle_save
)
from utils.data_manager import (
    TRAIN_TRACK_JSON, TEST_TRACK_JSON, DATA_DIR, RESULT_DIR,
    train_track_map, test_track_map
)
from object_tracking.library import VideoResult, TrackResult
from object_tracking.test.test_utils import calculate_iou, SAVE_DIR

sub_json_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-1/json_subject')
mask_save_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-1/attn_mask')

npy_save_dir = osp.join(mask_save_dir, 'npy_1')
vid_save_dir = osp.join(mask_save_dir, 'video_1')
os.makedirs(mask_save_dir, exist_ok=True)
os.makedirs(npy_save_dir, exist_ok=True)
os.makedirs(vid_save_dir, exist_ok=True)


def get_valid_coor(x, delta, xmax, xmin=0):
    x_new = x+delta
    x_new = min(xmax, max(xmin, x_new))
    return x_new

def get_box_center(box: list):
    return (box[0]+box[2]/2, box[1]+box[3]/2)

def get_velocity(center_0, center_1):
    return (center_1[0]-center_0[0], center_1[1]-center_0[1])

def expand_boxes(list_boxes: list, n: int=4, skip_frame: int=1):
    # xywh
    first_box, last_box = list_boxes[0], list_boxes[-1]
    list_center = [get_box_center(box) for box in list_boxes]
    first_velocity = get_velocity(list_center[skip_frame], list_center[0])
    last_velocity = get_velocity(list_center[-1 - skip_frame], list_center[-1])

    # Expand head
    new_head_boxes = []
    cur_box = first_box
    cur_center = list_center[0]
    for _ in range(n):
        w, h = cur_box[-2:]
        cur_center = (cur_center[0] + first_velocity[0], cur_center[1] + first_velocity[1])
        new_head_boxes.append([cur_center[0], cur_center[1], w, h])

    # Expand trail
    new_trail_boxes = []
    cur_box = last_box
    cur_center = list_center[-1]
    for _ in range(n):
        w, h = cur_box[-2:]
        cur_center = (cur_center[0] + last_velocity[0], cur_center[1] + last_velocity[1])
        new_trail_boxes.append([cur_center[0], cur_center[1], w, h])

    res = new_head_boxes[::-1] + list_boxes + new_trail_boxes
    return res


def interpolate_box(cur_idx, start_idx, end_idx, start_box, end_box):
    end_ratio = (end_idx-cur_idx)/(end_idx-start_idx)
    start_ratio = (cur_idx-start_idx)/(end_idx-start_idx)
    
    box1, box2 = start_box, end_box
    if isinstance(start_box, list):
        box1 = np.array(start_box) 
    if isinstance(end_box, list):
        box2 = np.array(end_box) 
    
    cur_box = start_ratio*box1 + end_ratio*box2 
    cur_box = cur_box.tolist()
    return cur_box

def generate_boxes(start_idx, end_idx, start_box, end_box):
    res = [] 
    for i in range(start_idx+1, end_idx):
        res.append(interpolate_box(i, start_idx, end_idx, start_box, end_box))
    return res

def refine_boxes(list_fids, list_boxes):
    N = len(list_fids)
    res = []
    latest_idx = 0
    for i in range(N-1):
        if list_fids[i+1] - list_fids[i] > 1:
            if calculate_iou(list_boxes[i+1], list_boxes[i]) < 0.2:
                new_boxes = generate_boxes(
                    list_fids[i], list_fids[i+1], list_boxes[i], list_boxes[i+1]
                )
                
                res += (list_boxes[latest_idx : i+1] + new_boxes)
                # print(f'len list_boxes: {len(list_boxes)}')
                # print(f'latest_idx: {latest_idx}')
                # print(f'i+1: {i+1}')
                # print(f'len split boxes: {len(list_boxes[latest_idx : i+1])}')
                # print(f'len(new_boxes): {len(new_boxes)}')
                # print(f'N box: {len(res)}')
                latest_idx = i+1
                pass # Interpolate box
    
    res += list_boxes[latest_idx:]
    return res

def get_mask_area(box, W, H, ratio=0.5):
    # xywh 
    x, y, w, h = box 
    exp_w = w*ratio
    exp_h = h*ratio

    new_x1 = int(get_valid_coor(x, -exp_w, W-1, 0))
    new_x2 = int(get_valid_coor(x, w+exp_w, W-1, 0))
    new_y1 = int(get_valid_coor(y, -exp_h, H-1, 0))
    new_y2 = int(get_valid_coor(y, h+exp_h, H-1, 0))

    return new_x1, new_y1, new_x2, new_y2

def get_frame_id(fname):
    fid = fname.split('.')[0].split('/')[-1]
    return int(fid)

def get_attn_mask(vid_data: dict, expand_ratio=0.35, n_expand=2):
    list_frames = vid_data['frames']
    list_fids = [get_frame_id(fname) for fname in list_frames]
    first_frame_path = list_frames[0]
    cv_frame = cv2.imread(osp.join(DATA_DIR, first_frame_path))
    H, W, C = cv_frame.shape

    mask = np.zeros(cv_frame.shape)
    list_boxes = vid_data['boxes']

    n_old = len(list_boxes)
    list_boxes = refine_boxes(list_fids, list_boxes)
    n_new = len(list_boxes)
    is_interpolate = False
    if n_new > n_old:
        is_interpolate = True

    skip_frame=3
    if len(list_boxes) >= skip_frame:
        exp_boxes = expand_boxes(list_boxes, n_expand)
    else:
        exp_boxes = list_boxes
    
    for box in exp_boxes:
        x1, y1, x2, y2 = get_mask_area(box, W, H, expand_ratio)
        mask[y1:y2, x1:x2, :] = 1.0

    return mask, is_interpolate

def main():
    VISUALIZE=False
    test_data = json_load(TEST_TRACK_JSON)
    # train_data = json_load(TRAIN_TRACK_JSON)
    list_keys = list(test_data.keys())
    for track_id in tqdm(list_keys):
        new_id = test_track_map[track_id]
        # if int(new_id) not in [100]:
        #     continue
        mask_save_path = osp.join(npy_save_dir, f'{new_id}.npy')
        vid_save_path = osp.join(vid_save_dir, f'{new_id}.avi')
        # if osp.isfile(mask_save_path):
        #     continue 

        track_json_path = osp.join(sub_json_dir, f'{new_id}.json')
        track_res = VideoResult(track_json_path)

        vid_data = test_data[track_id]
        vid_attn_mask, is_interpolate = get_attn_mask(vid_data)
        # vid_attn_mask = np.load(mask_save_path)
        np.save(mask_save_path, vid_attn_mask)
        # if is_interpolate:
        if VISUALIZE:
            track_res.visualize(vid_save_path, vid_attn_mask)

    pass

if __name__ == '__main__':
    main()