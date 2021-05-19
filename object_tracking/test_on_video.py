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

from deep_sort.deep_sort.iou_matching import iou

TRAIN_TRACK_DIR = "../classifier/data/Centernet2_train_veh_boxes.json"
TEST_TRACK_DIR = "../classifier/data/Centernet2_test_veh_boxes.json"

# ROOT_DIR = '/content/drive/MyDrive/THESIS/AI_CITY_2021/DATA/data_track_5/AIC21_Track5_NL_Retrieval'
# Use this below code when you have placed the dataset folder inside this project
ROOT_DIR = '../dataset'

TRAIN_FEAT_DIR = "../classifier/results/train_feat_tracking"
TEST_FEAT_DIR = "../classifier/results/test_feat_tracking"

SAVE_JSON_DIR = './results/annotate'
SAVE_VISUALIZE_DIR = './results/video'
os.makedirs(SAVE_JSON_DIR, exist_ok=True)
os.makedirs(SAVE_VISUALIZE_DIR, exist_ok=True)


def get_gt_from_idx(idx_image, gt_dict):
    frame_info = gt_dict[idx_image]
    key = list(frame_info.keys())[0]
    l = min(50, len(frame_info[key]))

    detections = []
    out_scores = []
    
    for i in range(l):
        x_0, y_0, x_1, y_1 = frame_info[key][i]
        x_0, y_0, x_1, y_1 = int(x_0), int(y_0), int(x_1), int(y_1)

        w = x_1 - x_0
        h = y_1 - y_0

        detections.append([x_0,y_0,w,h])
        out_scores.append(1)
    return detections, out_scores

def get_dict_track(filename):
    return json.load(open(filename))

def get_img_name(img_dict):
    ans = []
    l = len(img_dict)
    for i in range(l):
        name = list(img_dict[i].keys())[0]
        ans.append(name)
    return ans

def print_fail_dict(data, mode='VEHICLE'):
    print(f'{mode} fail features')
    for track_id in data.keys():
        print(f'{track_id}: {len(data[track_id])}')
    pass

def scan_data(track_keys, gt_dict):
    # Check extracted features (exist or not)
    fail_col, fail_veh = {}, {}

    for track_key in tqdm(track_keys):
        track_key = os.path.splitext(track_key)[0]
        
        fail_col[track_key], fail_veh[track_key] = [], []
        img_dict = gt_dict[track_key]
        img_names = get_img_name(img_dict)
        
        veh_path = os.path.join(VEH_DIR, f"{track_key}.pickle")
        with open(veh_path, 'rb') as handle:
            veh_features = pickle.load(handle)[track_key]
        
        color_path = os.path.join(COLOR_DIR, f'{track_key}.pickle')
        with open(color_path, 'rb') as handle:
            col_features = pickle.load(handle)[track_key]
        
        for img_name in img_names:
            img_col_feat, img_veh_feat = col_features.get(img_name), veh_features.get(img_name)
            if img_col_feat is None:
                fail_col[track_key].append(img_name)
            if img_veh_feat is None:
                fail_veh[track_key].append(img_name)

    col_fail_save_path = osp.join(SAVE_JSON_DIR, 'fail_col_feats.json')
    veh_fail_save_path = osp.join(SAVE_JSON_DIR, 'fail_veh_feats.json')
    json_dump(fail_col, col_fail_save_path)
    json_dump(fail_veh, veh_fail_save_path)
    print_fail_dict(fail_col)
    print_fail_dict(fail_veh)
    pass

def tracking(config):
    gt_dict = get_dict_track(config["track_dir"])
    track_keys = listdir(config["feat_dir"])
    print(f'>> Run DeepSort on {config["mode"]} mode')

    # track_keys = [
    #     "189bd009-a5db-4103-9edf-754126b34a42.pkl"
    # ]
    
    tracked_count = 0
    for track_key in track_keys:
        track_key = os.path.splitext(track_key)[0]
        img_dict = gt_dict[track_key]
        img_names = get_img_name(img_dict)
        
        feat_path = os.path.join(config["feat_dir"], f"{track_key}.pkl")
        with open(feat_path, 'rb') as handle:
            frame_feat = pickle.load(handle)

        ans = {}

        #Initialize deep sort.
        deepsort = deepsort_rbc()

        if config["save_video"]:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_visualize_path = os.path.join(SAVE_VISUALIZE_DIR, f'{track_key}.avi')
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
        
        save_json_path = os.path.join(SAVE_JSON_DIR, f'{track_key}.pkl')
        with open(save_json_path, 'wb') as handle:
            pickle.dump(ans, handle, protocol=pickle.HIGHEST_PROTOCOL)

        tracked_count += 1
        if tracked_count > 5:
            print("Complete tracking")
            break
    
    pass

def get_closest_box(list_boxes, target_box):
    new_list_boxes = [item.to_tlbr() for item in list_boxes]
    
    target_box = np.array(target_box)
    candidates = np.array(new_list_boxes)

    target_box[2:] -= target_box[:2]
    candidates[:, 2:] -= candidates[:, :2]
    
    scores = iou(target_box, candidates)
    best_id = np.argmax(scores)

    return new_list_boxes[best_id]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_video", action="store_true", help="Save video or not")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    config = {
        "train": {
            "track_dir": TRAIN_TRACK_DIR,
            "feat_dir": TRAIN_FEAT_DIR,
            "save_video": args.save_video,
            "mode": "train"
        },
        # "test": {
        #     "track_dir": TEST_TRACK_DIR,
        #     "feat_dir": TEST_FEAT_DIR,
        #     "save_video": args.save_video,
        #     "mode": "test"
        # },
    }
    for mode in config:
        tracking(config[mode])


