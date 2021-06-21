import os, json, pickle, h5py
import os.path as osp 
import cv2 
from tqdm import tqdm
from PIL import Image

import torch 
from torch import nn 
import torchvision
from classifier.library.box_extractor import init_model
from classifier.utils import preprocess_input, get_feat_from_subject_box
from classifier.utils.config import cfg_veh, cfg_col

## GLOBAL VARIABLES
# ROOT_DIR = '/content/AI_CITY_2021/DATA/data_track_5/AIC21_Track5_NL_Retrieval'
ROOT_DIR = '/scratch/ntphat/dataset'

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

MODEL_NAME = 'resnet152'
model = torchvision.models.resnet152(pretrained=True)
model.fc = Identity()
model = model.cuda()
model.eval()

print('Loaded global extractor successfully')
DATA_DIR = osp.join(ROOT_DIR, 'data')
TRAIN_TRACK_JSON = osp.join(DATA_DIR, 'train-tracks.json')
TEST_TRACK_JSON = osp.join(DATA_DIR, 'test-tracks.json')

# Save result to Retrieval model data
RESULT_DIR = '../retrieval_model/data'
DATASET_NAME = 'aic21'
FEAT_SAVE_DIR = osp.join(RESULT_DIR, DATASET_NAME)
GLOBAL_SIZE = 224

os.makedirs(FEAT_SAVE_DIR, exist_ok=True)
train_track = json.load(open(TRAIN_TRACK_JSON))
test_track = json.load(open(TEST_TRACK_JSON))
data_track = {'train': train_track, 'test': test_track}

veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True)
veh_model = veh_model.cuda()
col_model = col_model.cuda()

print(f'Loaded box extractor successfully')

@torch.no_grad()
def extract_feature(data_track, img_dir, split='train'):
    feats = {}
    for qid in tqdm(data_track):
        track_feats = []
        if split is not None:
            first_frame = data_track[qid]['frames'][0]
            infos = first_frame.split('/')
            track_split = infos[0]
            if track_split != split:
                continue
                
        for frame_path, box_coor in zip(data_track[qid]['frames'], data_track[qid]['boxes']):
            # img_split = get_img_split(frame_path)
            img_path = osp.join(img_dir, frame_path)
            cv_img = cv2.imread(img_path)
            img_h, img_w, img_c = cv_img.shape
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            inp = preprocess_input(cv_img)
            global_feat = model(inp.unsqueeze(0).cuda()) # [2048]
            
            box_coor = [int(x) for x in box_coor]
            x, y, w, h = box_coor
            crop = cv_img[y:y+h, x:x+w, :]
            box_feat = get_feat_from_subject_box(crop, veh_model, col_model) # [2*2048] 
            
            feat = torch.cat((global_feat.squeeze().cpu(), box_feat)) #[6144]
            coor_feat = torch.Tensor([x/img_w, y/img_h, w/img_w, h/img_h]).float()
            feat = torch.cat((feat, coor_feat)) # [6148]

            track_feats.append(feat.unsqueeze(0))
        
        track_feats = torch.cat(track_feats, dim=0)
        feats[qid] = track_feats.detach().numpy()

    return feats

def pickle_save(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'save result to {save_path}')

def h5_save(data, save_path):
    data_h5 = h5py.File(save_path, "w")
    for qid in data:
        data_h5[qid] = data[qid]
    data_h5.close()
    print(f'save result to {save_path}')

def main():
    SPLIT = None #sys.argv[1] # 'validation'
    for mode in ['train', 'test']:
        split_img_dir = ROOT_DIR
        if SPLIT is None:
            print(f'[{mode}]: Extract feature')
            feat_save_path = osp.join(FEAT_SAVE_DIR, f'{mode}_{MODEL_NAME}-{GLOBAL_SIZE}_6148.pkl')
        else:
            print(f'[{mode}]: Extract feature for {SPLIT} split')
            feat_save_path = osp.join(FEAT_SAVE_DIR, f'{mode}_{MODEL_NAME}-{GLOBAL_SIZE}_6148_{SPLIT}.pkl')
        h5_save_path = feat_save_path.replace('.pkl', '.h5')

        feats = extract_feature(data_track[mode], split_img_dir, SPLIT)
        pickle_save(feats, feat_save_path)
        h5_save(feats, h5_save_path)

    
if __name__ == '__main__':
    main()
