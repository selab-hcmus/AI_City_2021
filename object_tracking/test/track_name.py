"""
Get vehicle predictions for each tracks
"""

import os 
import os.path as osp
from tqdm import tqdm

from utils import json_load, json_save
from object_tracking.library import VideoResult, TrackResult
from object_tracking.test.test_utils import SAVE_DIR

from classifier import ClassifierManager

track_res_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-3', 'json_stop')
save_dir = osp.join(SAVE_DIR, 'test_deepsort_v4-June14_15h00')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(track_res_dir, exist_ok=True)


def main():
    det_manager = ClassifierManager(cuda=True, load_ckpt=True, eval=True)
    list_files = os.listdir(track_res_dir)
    vid_save_dir = osp.join(save_dir, 'visualize') # Change vis save dir here
    os.makedirs(vid_save_dir, exist_ok=True)
    for fname in tqdm(list_files):
        fpath = osp.join(track_res_dir, fname)
        vid_data = VideoResult(fpath)
        vid_data.set_class_names(det_manager)

        vid_id = fname.split('.')[0]
        vid_save_path = osp.join(vid_save_dir, f'{vid_id}.avi')
        if osp.isfile(vid_save_path):
            continue
        vid_data.visualize(vid_save_path)
        pass
    pass

if __name__ == '__main__':
    main()