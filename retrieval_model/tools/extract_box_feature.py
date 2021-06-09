import os, sys, json, pickle
import os.path as osp 
import cv2 
from tqdm import tqdm
from PIL import Image

import torch 
from torch import nn 
import torchvision
from torchvision import datasets, models, transforms

from box_extractor import init_model, preprocess_input

ROOT_DIR = '/home/ntphat/projects/coot-aic'
IMAGE_DIR = '/home/ntphat/projects/aic21/aic2021/data'
BOX_JSON_PATH = osp.join(ROOT_DIR, 'data/Centernet2_train_veh_boxes.json')
box_dict = json.load(open(BOX_JSON_PATH))
cfg_veh = {
    'MODEL': 'efficientnet-b5',
    'NUM_CLASSES': 6,
    'WEIGHT': '/home/ntphat/projects/aic21/aic2021/weights/box_class/veh_best_model.pt',
    'SAVE_FEATURE_DIR': '/home/ntphat/projects/coot-aic/data/box_feats/vehicle',
}
cfg_col = {
    'MODEL': 'efficientnet-b5',
    'NUM_CLASSES': 8,
    'WEIGHT': '/home/ntphat/projects/aic21/aic2021/weights/box_class/col_best_model.pt',
    'SAVE_FEATURE_DIR': '/home/ntphat/projects/coot-aic/data/box_feats/color',
}

veh_model, col_model = init_model(cfg_veh, cfg_col)
veh_model = veh_model.cuda()
col_model = col_model.cuda()

print(f'>> Init box extractor successfully')

def check_refined_boxes():
    fail_boxes = {}

    for track_id in tqdm(box_dict.keys()):
        fail_boxes[track_id] = []
        for frame_data in box_dict[track_id]:
            frame_id, frame_boxes = list(frame_data.items())[0]
            if len(frame_boxes) == 0:
                fail_boxes[track_id].append(frame_id)

    for track_id in fail_boxes.keys():
        if len(fail_boxes[track_id]) != 0:
            print(f'{track_id} number of frames with no boxes: {len(fail_boxes[track_id])}')        
    

def extract_box_feat(model, cfg, mode: str):
    os.makedirs(cfg['SAVE_FEATURE_DIR'], exist_ok=True)

    fail_result = {}

    for track_id in tqdm(box_dict.keys()):
        ans = {track_id: {}}
        fail_result[track_id] = []
        filename = f'{track_id}.pickle'
        save_path = os.path.join(cfg['SAVE_FEATURE_DIR'], filename)
        
        for val in (box_dict[track_id]):
            val_keys = list(val.keys())

            for key in val_keys:
                ans[track_id][key] = []
                new_path = os.path.join(IMAGE_DIR, key)
                img = cv2.imread(new_path)
                for box in val[key]:
                    x_0, y_0, x_1, y_1 = box
                    x_0, y_0, x_1, y_1 = int(x_0), int(y_0), int(x_1), int(y_1)
                    crop_img = img[y_0:y_1, x_0:x_1, :]
                    crop_img = preprocess_input(Image.fromarray(crop_img).convert('RGB')).cuda()
                    crop_img = crop_img.unsqueeze(0)
                    with torch.no_grad():
                        feature = model.extract_feature(crop_img)
                    ans[track_id][key].append(feature)

            if len(ans[track_id][key]) == 0:
                fail_result[track_id].append(key)
                

        with open(save_path, 'wb') as handle:
            pickle.dump(ans, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'/home/ntphat/projects/coot-aic/data/box_feats/fail_extraction_{mode}.json', 'w') as handle:
        json.dump(ans, handle, indent=2)

if __name__ == '__main__':
    # check_refined_boxes()

    print('Extract Vehicle feature')
    extract_box_feat(veh_model, cfg_veh, 'veh')
    print(f'save result to {cfg_veh["SAVE_FEATURE_DIR"]}')

    # print('Extract Color feature')
    # extract_box_feat(col_model, cfg_col, 'col')
    print(f'save result to {cfg_col["SAVE_FEATURE_DIR"]}')

    pass