import os 
import os.path as osp
import cv2 
import numpy as np
from tqdm import tqdm 

import torch 
import torch.nn as nn
from object_tracking.library import VideoResult, TrackResult
from dataset.data_manager import DATA_DIR

SAVE_DIR = '/home/ntphat/projects/AI_City_2021/object_tracking/results_exp'
exp_id = 'test_deepsort_v4-1'
subject_json_dir = osp.join(SAVE_DIR, exp_id, 'json_stop')

save_dir = osp.join(SAVE_DIR, exp_id, 'attn_mask')
vid_save_dir = osp.join(save_dir, 'video')
npy_save_dir = osp.join(save_dir, 'npy')
os.makedirs(vid_save_dir, exist_ok=True)
os.makedirs(npy_save_dir, exist_ok=True)

## PARAMETER
delta_w = 10
delta_h = 10

def expand_mask_pooling(road_mask, n_loop=4, cuda=True):
    layer = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    mask = np.expand_dims(road_mask, axis=0)
    mask_tensor = torch.Tensor(mask)
    
    if len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(1)
    
    if torch.cuda.is_available() and cuda:
        layer = layer.cuda()
        mask_tensor = mask_tensor.cuda()

    for count in range(n_loop):
        mask_tensor = layer(mask_tensor)
    
    if torch.cuda.is_available() and cuda:
        mask_tensor = mask_tensor.cpu()

    result = mask_tensor.squeeze().numpy()
    return result

def get_valid_coor(x, delta_x, max_x, min_x=0):
    new_x = x + delta_x
    new_x = max(min(new_x, max_x), min_x)
    return new_x

def create_attn_mask(vid_data, vis_save_path=None, vis_save_dir=None):
    if vid_data.subject is None:
        print(f'Video has no subject')
        return
    sub_data = vid_data.track_map[vid_data.subject]
    first_frame = vid_data.list_frames[0]
    first_img = cv2.imread(osp.join(DATA_DIR, first_frame))
    H, W, C = first_img.shape

    attn_mask = np.zeros(first_img.shape)

    for box in sub_data.boxes:
        x1, y1, x2, y2 = box 
        box_w, box_h = x2-x1, y2-y1 
        x1_box = int(get_valid_coor(x1, - 3*box_w/5, W-1, 0))
        y1_box = int(get_valid_coor(y1, - 3*box_h/5, H-1, 0))
        x2_box = int(get_valid_coor(x2, + 3*box_w/5, W-1, 0))
        y2_box = int(get_valid_coor(y2, + 3*box_h/5, H-1, 0))
        attn_mask[y1_box : y2_box, x1_box : x2_box, :] = 1.0

    # cv2.imwrite(osp.join(vis_save_dir, 'before_expand.png'), (attn_mask*255).astype(np.uint8))
    # kernel_size = 8
    # gauss_kernel = cv2.getGaussianKernel(kernel_size, sigma=1.6)
    # attn_mask = cv2.dilate(attn_mask, gauss_kernel)
    # cv2.imwrite(osp.join(vis_save_dir, 'after_dilate.png'), (attn_mask*255).astype(np.uint8))
    # attn_mask = expand_mask_pooling(attn_mask, n_loop=15, cuda=True)
    attn_mask = cv2.GaussianBlur(attn_mask, (5, 5), 1.4)
    # cv2.imwrite(osp.join(vis_save_dir, 'after_expand.png'), (attn_mask*255).astype(np.uint8))
    return attn_mask

def remove_obj_out_of_mask(vid_data, attn_mask):
    pass

def get_track_id_from_fname(fname: str):
    # fname: 'track_id.json'
    return fname.split('.')[0]

def main():
    for fname in tqdm(os.listdir(subject_json_dir)):
        fpath = osp.join(subject_json_dir, fname)
        vid_data = VideoResult(fpath)
        track_id = get_track_id_from_fname(fname)
        vis_save_path = osp.join(vid_save_dir, f'{track_id}.avi')

        attn_mask = create_attn_mask(vid_data, vis_save_path, vid_save_dir) # shape = ...
        vid_data.visualize(vis_save_path, attn_mask)
        np.save(osp.join(npy_save_dir, f'{track_id}.npy'), attn_mask)

        # remove_obj_out_of_mask(vid_data, attn_mask)
        # Save ? 

    pass

if __name__ == '__main__':
    main()
