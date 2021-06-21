import json, pickle
import os 
import os.path as osp 
import cv2
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image

import torch 
from torch import nn 
import torchvision
from torchvision import transforms

from utils import prepare_dir, dict_load, dict_save
from utils.data_manager import (
    train_track_map, test_track_map, 
    TRAIN_TRACK_JSON, TEST_TRACK_JSON, DATA_DIR
)
from object_tracking.utils import SAVE_DIR
import torchreid
from torchreid.utils import FeatureExtractor
from config import (
    MODEL_NAME, WEIGHT_DIR
)

global_extractor = torchvision.models.resnet152(pretrained=True)
global_extractor.fc = nn.Identity()
global_extractor = global_extractor.cuda()
global_extractor.eval()

IMAGE_SIZE = (224,224)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def global_preprocess_list_imgs(list_imgs):
    images = []
    for element in list_imgs:
        if isinstance(element, np.ndarray):
            element = Image.fromarray(element).convert('RGB')
        element = val_transform(element)
        images.append(element)
    
    return torch.stack(images, dim=0).cuda()
    

BATCH_SIZE = 64
extractor = FeatureExtractor(
    model_name=MODEL_NAME,
    model_path=WEIGHT_DIR,#osp.join(WEIGHT_DIR, MODEL_NAME, 'model.pth.tar-25'),
    device='cuda'
)

save_dir = osp.join(DATA_DIR, 'retrieval_model', 'video_feat')
save_dir = {
    'train': prepare_dir(osp.join(save_dir, f'train_{MODEL_NAME}')),
    'test': prepare_dir(osp.join(save_dir, f'test_{MODEL_NAME}')),

    'train_global': prepare_dir(osp.join(save_dir, f'train_resnet152')),
    'test_global': prepare_dir(osp.join(save_dir, f'test_resnet152')),

    'train_box': prepare_dir(osp.join(save_dir, f'train_box')),
    'test_box': prepare_dir(osp.join(save_dir, f'test_box')),
}
SAVE_PERIOD = 10

train_track = dict_load(TRAIN_TRACK_JSON)
test_track = dict_load(TEST_TRACK_JSON)
data_track = {'train': train_track, 'test': test_track}


def extract_box(data_track, mode_save_dir: str):
    count = 1
    list_keys = list(data_track.keys())
    print(f'Extract {len(list_keys)} tracks')

    for i, key_track in tqdm(enumerate(list_keys), total=len(list_keys)):
        count += 1
        track_save_path = osp.join(mode_save_dir, f'{key_track}.pkl')
        if osp.isfile(track_save_path):
            continue 
        
        list_frames = data_track[key_track]['frames']
        list_boxes = data_track[key_track]['boxes']
        
        track_feat = {}
        boxes2feed = []
        for fname, box in zip(list_frames, list_boxes):
            fpath = osp.join(DATA_DIR, fname)
            cv_img = cv2.imread(fpath)
            H, W, C = cv_img.shape
            box = [int(x) for x in box]
            x, y, w, h = box
            track_feat[fname] = np.array([x/W, y/H, (x+w)/W, (y+h)/H, (w*h)/(W*H)])
            
        # print(f'Extract {count}th')
        dict_save(track_feat, track_save_path)
        # feat[key_track] = track_feat

    pass

def extract_feature_global(data_track, mode_save_dir: str):
    count = 1
    list_keys = list(data_track.keys())
    print(f'Extract {len(list_keys)} tracks')

    for i, key_track in tqdm(enumerate(list_keys), total=len(list_keys)):
        count += 1
        track_save_path = osp.join(mode_save_dir, f'{key_track}.pkl')
        if osp.isfile(track_save_path):
            continue 
        
        list_frames = data_track[key_track]['frames']
        list_boxes = data_track[key_track]['boxes']
        
        track_feat = {}
        boxes2feed = []
        for fname, box in zip(list_frames, list_boxes):
            fpath = osp.join(DATA_DIR, fname)
            cv_img = cv2.imread(fpath)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            box = [int(x) for x in box]
            x, y, w, h = box
            crop = cv_img[y:y+h, x:x+w, :]

            boxes2feed.append(crop)

        n_box = len(boxes2feed)
        if n_box > BATCH_SIZE:
            list_subfeets = []
            n_split = (n_box-1)//BATCH_SIZE 
            count = 0
            for j in range(n_split):
                max_id = (j+1)*n_split
                split_boxes = boxes2feed[count: max_id]
                split_boxes = global_preprocess_list_imgs(split_boxes)
                sub_feet = extractor(split_boxes).detach().cpu().numpy()
                list_subfeets.append(sub_feet)
                count = max_id
                pass

            if count < n_box:
                split_boxes = boxes2feed[count:]
                split_boxes = global_preprocess_list_imgs(split_boxes)
                sub_feet = extractor(split_boxes).detach().cpu().numpy()
                list_subfeets.append(sub_feet)

            track_feed = np.concatenate(list_subfeets, axis=0)
            assert track_feed.shape[0] == n_box, f"Fail when too many boxes, track_feed: {track_feed.shape}, n_box: {n_box}"
        
        else:
            boxes2feed = global_preprocess_list_imgs(boxes2feed)
            track_feed = extractor(boxes2feed).detach().cpu().numpy()
        
        if i == 0:
            print(f'First track feat shape: {track_feed.shape}')

        for j, fname in enumerate(list_frames):
            track_feat[fname] = track_feed[j]
            pass

        # print(f'Extract {count}th')
        dict_save(track_feat, track_save_path)
        # feat[key_track] = track_feat

    pass

def extract_feature(data_track, mode_save_dir: str):
    count = 1
    list_keys = list(data_track.keys())
    print(f'Extract {len(list_keys)} tracks')

    for i, key_track in tqdm(enumerate(list_keys), total=len(list_keys)):
        count += 1
        track_save_path = osp.join(mode_save_dir, f'{key_track}.pkl')
        if osp.isfile(track_save_path):
            continue 
        
        list_frames = data_track[key_track]['frames']
        list_boxes = data_track[key_track]['boxes']
        
        track_feat = {}
        boxes2feed = []
        for fname, box in zip(list_frames, list_boxes):
            fpath = osp.join(DATA_DIR, fname)
            cv_img = cv2.imread(fpath)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

            box = [int(x) for x in box]
            x, y, w, h = box
            crop = cv_img[y:y+h, x:x+w, :]

            boxes2feed.append(crop)

        n_box = len(boxes2feed)
        if n_box > BATCH_SIZE:
            list_subfeets = []
            n_split = (n_box-1)//BATCH_SIZE 
            count = 0
            for j in range(n_split):
                max_id = (j+1)*n_split
                split_boxes = boxes2feed[count: max_id]
                sub_feet = extractor(split_boxes).detach().cpu().numpy()
                list_subfeets.append(sub_feet)
                count = max_id
                pass

            if count < n_box:
                split_boxes = boxes2feed[count:]
                sub_feet = extractor(split_boxes).detach().cpu().numpy()
                list_subfeets.append(sub_feet)

            track_feed = np.concatenate(list_subfeets, axis=0)
            assert track_feed.shape[0] == n_box, f"Fail when too many boxes, track_feed: {track_feed.shape}, n_box: {n_box}"
        
        else:
            track_feed = extractor(boxes2feed).detach().cpu().numpy()
        
        if i == 0:
            print(f'First track feat shape: {track_feed.shape}')

        for j, fname in enumerate(list_frames):
            track_feat[fname] = track_feed[j]
            pass

        # print(f'Extract {count}th')
        dict_save(track_feat, track_save_path)
        # feat[key_track] = track_feat

    pass

def test():
    feat_path = '/home/ntphat/projects/AI_City_2021/object_tracking/reid/results/train_feat_tracking/0.pkl'
    feat = None
    with open(feat_path, 'rb') as f:
        feat = pickle.load(f)

    for k in feat:
        print(f'{k}: {feat[k].shape}')

if __name__ == '__main__':
    # test()

    # for mode in ["train", "test"]:
    # for mode in ["train_global", "test_global"]:
    for mode in ["train_box", "test_box"]:
        print(f"Extract in {mode} data")
        print(f'Save result to {save_dir[mode]}')

        extract_box(data_track[mode.split('_')[0]], save_dir[mode])
        

    