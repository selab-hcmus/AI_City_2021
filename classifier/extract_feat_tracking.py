import json, pickle
import os.path as osp 
import cv2 
from tqdm import tqdm

import torch 
from box_extractor import init_model
from config import cfg_veh, cfg_col
from utils import get_feat_from_subject_box

## GLOBAL VARIABLES

## Use this below code when you have placed the dataset folder inside this project
# ROOT_DIR = '../dataset'
ROOT_DIR = '/content/AIC21_Track5_NL_Retrieval'

SAVE_DIR = './data/results'

TRAIN_TRACK_JSON = './data/Centernet2_train_veh_boxes.json'
TEST_TRACK_JSON = './data/Centernet2_test_veh_boxes.json'

train_track = json.load(open(TRAIN_TRACK_JSON))
test_track = json.load(open(TEST_TRACK_JSON))
data_track = {'train': train_track, 'test': test_track}

veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True)
veh_model = veh_model.cuda()
col_model = col_model.cuda()

@torch.no_grad()
def extract_feature(data_track, data_dir):
    feat = {}
    for key_track in tqdm(data_track):
        track_feat = {}
        for frame_dict in data_track[key_track]:
            frame_path = list(frame_dict.keys())[0]
            frame_feat = []
            for box_coor in frame_dict[frame_path]:
                img_path = osp.join(data_dir, frame_path)
                cv_img = cv2.imread(img_path)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
                box_coor = [int(x) for x in box_coor]
                x_0, y_0, x_1, y_1 = box_coor
                crop = cv_img[y_0:y_1, x_0:x_1, :]
                box_feat = get_feat_from_subject_box(crop, veh_model, col_model)
                box_feat = box_feat.detach().numpy()
                frame_feat.append(box_feat)
            track_feat[frame_path] = frame_feat
        feat[key_track] = track_feat
    return feat

def pickle_save(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'save result to {save_path}')
    
if __name__ == '__main__':
    for mode in ["train", "test"]:
        print(f"Extract in {mode} data")
        feat = extract_feature(data_track[mode], ROOT_DIR)
        pickle_save(feat, SAVE_DIR)