import numpy as np
import torch
import cv2, pickle
import os
from os import listdir
import os.path as osp 
from deepsort import deepsort_rbc
import json
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

from deep_sort.iou_matching import iou
from dataset.data_manager import test_track_map, train_track_map
from utils import(
    get_gt_from_idx, get_dict_track, get_img_name, print_fail_dict, get_closest_box
)
from tools import (
    convert_video_track, visualize
)

TRAIN_TRACK_DIR = "../classifier/data/Centernet2_train_veh_order.json"
TEST_TRACK_DIR = "../classifier/data/Centernet2_test_veh_order.json"

# ROOT_DIR = '/content/drive/MyDrive/THESIS/AI_CITY_2021/DATA/data_track_5/AIC21_Track5_NL_Retrieval'
# Use this below code when you have placed the dataset folder inside this project
ROOT_DIR = '../dataset'
TRAIN_FEAT_DIR = "../classifier/results/train_feat_tracking"
TEST_FEAT_DIR = "../classifier/results/test_feat_tracking"

SAVE_JSON_DIR = './results/annotate'
SAVE_VISUALIZE_DIR = './results/video'
os.makedirs(SAVE_JSON_DIR, exist_ok=True)
os.makedirs(SAVE_VISUALIZE_DIR, exist_ok=True)

NUM_TO_EXP = 5
ID_MAP = {'train': train_track_map, 'test': test_track_map}

def tracking(config, json_save_dir: str, vis_save_dir: str, verbose=True):
    mode_json_dir = json_save_dir
    
    gt_dict = get_dict_track(config["track_dir"])
    track_keys = listdir(config["feat_dir"])
    print(f'>> Run DeepSort on {config["mode"]} mode, save result to {mode_json_dir}')

    # track_keys = [
    #     "189bd009-a5db-4103-9edf-754126b34a42.pkl"
    # ]
    
    tracked_count = 0
    for track_key in track_keys[:NUM_TO_EXP]:
        track_key = os.path.splitext(track_key)[0]
        # track_order = ID_MAP[mode][track_key]
        track_order = track_key

        if verbose:
            print(f'tracking order {track_order}')
        img_dict = gt_dict[track_key]
        img_names = get_img_name(img_dict)
        
        feat_path = os.path.join(config["feat_dir"], f"{track_order}.pkl")
        with open(feat_path, 'rb') as handle:
            frame_feat = pickle.load(handle)

        ans = {}

        #Initialize deep sort.
        deepsort = deepsort_rbc()

        if config["save_video"]:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_visualize_path = os.path.join(vis_save_dir, f'{track_order}.avi')
            # out = cv2.VideoWriter(save_visualize_path,fourcc, 2, (1920,1080))
            out = None

        for i in tqdm(range(len(img_names))):
            img_path = os.path.join(ROOT_DIR, img_names[i])

            frame = cv2.imread(img_path)
            frame = frame.astype(np.uint8)
            
            if i == 0 and config["save_video"]:
                h, w, c = frame.shape
                out = cv2.VideoWriter(save_visualize_path,fourcc, 1, (w,h))

            detections, out_scores = get_gt_from_idx(i, gt_dict[track_key])

            detections = np.array(detections)
            out_scores = np.array(out_scores)
            
            features = frame_feat[img_names[i]]
            
            tracker, detections_class = deepsort.run_deep_sort(out_scores, detections, features)

            track_list = []
            count = 0

            for track in tracker.tracks:
                count += 1
                if not track.is_confirmed() or track.time_since_update > 2:
                    continue

                bbox = track.to_tlbr() #Get the corrected/predicted bounding box
                id_num = str(track.track_id) #Get the ID for the particular track.
                features = track.features #Get the feature vector corresponding to the detection.

                track_dict = {}
                track_dict["id"] = id_num

                ans_box = get_closest_box(detections_class, bbox)
                bbox = ans_box
                # track_dict["box"] = [int(ans_box[0]), int(ans_box[1]), int(ans_box[2]), int(ans_box[3])]
                track_dict["box"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                track_dict["feature"] = features
                track_list.append(track_dict)

                if config["save_video"]:
                    #Draw bbox from tracker.
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                    #Draw bbox from detector. Just to compare.
                    for det in detections_class:
                        bbox = det.to_tlbr()
                        cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
            
            ans[img_names[i]] = track_list

            if config["save_video"]:
                out.write(frame)
        
        if config["save_video"]:
            out.release()
        
        save_json_path = os.path.join(mode_json_dir, f'{track_order}.json')
        reformat_res = convert_video_track(ans, save_json_path)

    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video", action="store_true", help="Save video or not")
    parser.add_argument("--exp_id", type=str, default='v1')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = {
        # "train": {
        #     "track_dir": TRAIN_TRACK_DIR,
        #     "feat_dir": TRAIN_FEAT_DIR,
        #     "save_video": args.save_video,
        #     "mode": "train"
        # },
        "test": {
            "track_dir": TEST_TRACK_DIR,
            "feat_dir": TEST_FEAT_DIR,
            "save_video": args.save_video,
            "mode": "test"
        },
    }
    SAVE_DIR = '/home/ntphat/projects/AI_City_2021/object_tracking/results'
    exp_save_dir = osp.join(SAVE_DIR, f'Exp_{args.exp_id}')
    os.makedirs(exp_save_dir, exist_ok=True)
    json_save_dir = osp.join(exp_save_dir, 'json')
    vid_save_dir = osp.join(exp_save_dir, 'video')
    os.makedirs(json_save_dir, exist_ok=True)
    os.makedirs(vid_save_dir, exist_ok=True)

    for mode in config:
        print(f'Run on mode {mode}')
        tracking(config[mode], json_save_dir, vid_save_dir)


