import json, os
import os.path as osp 
import cv2 
from tqdm import tqdm

import torch 
from classifier.box_extractor import init_model
from classifier.config import cfg_veh, cfg_col
from classifier.utils import get_feat_from_subject_box, get_feat_from_model

from utils import pickle_save, json_load
from dataset.data_manager import test_track_map, train_track_map

## GLOBAL VARIABLES
# ROOT_DIR = '/content/AIC21_Track5_NL_Retrieval'
# ROOT_DIR = '/scratch/ntphat/dataset'
## Use this below code when you have placed the dataset folder inside this project
ROOT_DIR = '/home/ntphat/projects/AI_City_2021/dataset'

# SAVE_DIR = '/scratch/ntphat/results'
MODEL_ID = 'May31_uptrain'
SAVE_DIR = f'./results/{MODEL_ID}'

TRAIN_TRACK_JSON = './data/Centernet2_train_veh_boxes.json'
TEST_TRACK_JSON = './data/Centernet2_test_veh_boxes.json'

cfg_veh['WEIGHT'] = osp.join('./results', MODEL_ID, 'veh_best_model.pt')
os.makedirs(SAVE_DIR, exist_ok=True)

data_track = {'train': json_load(TRAIN_TRACK_JSON), 'test': json_load(TEST_TRACK_JSON)}
data_key_map = {'train': train_track_map, 'test': test_track_map}

veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True, eval=True)
veh_model = veh_model.cuda()
col_model = col_model.cuda()


@torch.no_grad()
def extract_feature(data_track, key_map, data_dir, mode_save_dir: str):
    count = 1
    list_keys = list(data_track.keys())
    for key_track in tqdm(data_track, total=len(list_keys)):
        count += 1
        order = key_map[key_track]
        track_save_path = osp.join(mode_save_dir, f'{order}.pkl')
        if osp.isfile(track_save_path):
            continue 

        track_feat = {}
        for frame_dict in data_track[key_track]:
            frame_path = list(frame_dict.keys())[0]
            list_boxes = []
            
            for box_coor in frame_dict[frame_path]:
                img_path = osp.join(data_dir, frame_path)
                cv_img = cv2.imread(img_path)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                box_coor = [int(x) for x in box_coor]
                x_0, y_0, x_1, y_1 = box_coor
                crop = cv_img[y_0:y_1, x_0:x_1, :]
                list_boxes.append(crop)
                
                # box_feat = get_feat_from_subject_box(crop, veh_model, col_model)
                # box_feat = box_feat.detach().numpy()
                # frame_feat.append(box_feat)
            
            col_feat = get_feat_from_model(list_boxes, col_model)
            veh_feat = get_feat_from_model(list_boxes, veh_model)
            track_feat[frame_path] = torch.cat([veh_feat, col_feat], dim=1).detach().cpu().numpy()
        
        # print(f'Extract {count}th')
        pickle_save(track_feat, track_save_path)
    pass

if __name__ == '__main__':
    # for mode in ["train", "test"]:
    for mode in ["test"]:
        print(f"Extract in {mode} data")
        mode_save_dir = osp.join(SAVE_DIR, f'{mode}_feat_tracking')
        os.makedirs(mode_save_dir, exist_ok=True)
        extract_feature(data_track[mode], data_key_map[mode], ROOT_DIR, mode_save_dir)
