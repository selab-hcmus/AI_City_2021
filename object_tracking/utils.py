import json
import os 
import os.path as osp
from tqdm import tqdm
import numpy as np
from object_tracking.deep_sort.iou_matching import iou

TRAIN_TRACK_DIR = "/home/ntphat/projects/AI_City_2021/classifier/data/Centernet2_train_veh_order.json"
TEST_TRACK_DIR = "/home/ntphat/projects/AI_City_2021/classifier/data/Centernet2_test_veh_order.json"
VEHCOL_FEAT_DIR = "/home/ntphat/projects/AI_City_2021/classifier/results/train_feat_tracking"
REID_FEAT_DIR = "reid/results/train_feat_tracking"
ROOT_DIR = '/home/ntphat/projects/AI_City_2021/dataset'

TRAIN_TRACKING_RESULT = './results/annotate_time_train'
TEST_TRACKING_RESULT = './results/annotate_time_test'

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


# def scan_data(track_keys, gt_dict):
#     # Check extracted features (exist or not)
#     fail_col, fail_veh = {}, {}

#     for track_key in tqdm(track_keys):
#         track_key = os.path.splitext(track_key)[0]
        
#         fail_col[track_key], fail_veh[track_key] = [], []
#         img_dict = gt_dict[track_key]
#         img_names = get_img_name(img_dict)
        
#         veh_path = os.path.join(VEH_DIR, f"{track_key}.pickle")
#         with open(veh_path, 'rb') as handle:
#             veh_features = pickle.load(handle)[track_key]
        
#         color_path = os.path.join(COLOR_DIR, f'{track_key}.pickle')
#         with open(color_path, 'rb') as handle:
#             col_features = pickle.load(handle)[track_key]
        
#         for img_name in img_names:
#             img_col_feat, img_veh_feat = col_features.get(img_name), veh_features.get(img_name)
#             if img_col_feat is None:
#                 fail_col[track_key].append(img_name)
#             if img_veh_feat is None:
#                 fail_veh[track_key].append(img_name)

#     col_fail_save_path = osp.join(SAVE_JSON_DIR, 'fail_col_feats.json')
#     veh_fail_save_path = osp.join(SAVE_JSON_DIR, 'fail_veh_feats.json')
#     json_dump(fail_col, col_fail_save_path)
#     json_dump(fail_veh, veh_fail_save_path)
#     print_fail_dict(fail_col)
#     print_fail_dict(fail_veh)
#     pass
