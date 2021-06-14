import json
import os 
import os.path as osp
from tqdm import tqdm
import numpy as np

from utils.data_manager import RESULT_DIR
from object_tracking.deep_sort.iou_matching import iou

SAVE_DIR = osp.join(RESULT_DIR, 'object_tracking')
EXP_SAVE_DIR = osp.join(RESULT_DIR, 'object_tracking_exp')
TRAIN_TRACK_DIR = "/content/AI_City_2021/classifier/data/Centernet2_train_veh_order.json"
TEST_TRACK_DIR = "/content/AI_City_2021/classifier/data/Centernet2_test_veh_order.json"

VEHCOL_FEAT_DIR = "/content/AI_City_2021/classifier/results/train_feat_tracking"
REID_FEAT_DIR = "reid/results/train_feat_tracking"
ROOT_DIR = '/content/AI_City_2021/dataset'

def get_closest_box(list_boxes, target_box):
    new_list_boxes = [item.to_tlbr() for item in list_boxes]
    
    target_box = np.array(target_box)
    candidates = np.array(new_list_boxes)

    target_box[2:] -= target_box[:2]
    candidates[:, 2:] -= candidates[:, :2]
    
    scores = iou(target_box, candidates)
    best_id = np.argmax(scores)

    return new_list_boxes[best_id]

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

def json_dump(data: dict, save_path: str):
    with open(save_path, 'r') as f:
        json.dump(data, f, indent=2)
    pass

