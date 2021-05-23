"""Get class prediction for tracking feature
"""

import os, cv2, pickle, json 
import os.path as osp

import torch

TRACK_FEAT_DIR = '../object_tracking/results'
TRAIN_FEAT_DIR = osp.join(TRACK_FEAT_DIR, 'annotate_train')
TEST_FEAT_DIR = osp.join(TRACK_FEAT_DIR, 'annotate_test')

from box_extractor import init_model
from config import cfg_veh, cfg_col
from utils import get_feat_from_subject_box, pickle_save, pickle_load

veh_model, col_model = init_model(cfg_veh, cfg_col, load_ckpt=True, eval=True)
veh_model = veh_model.cuda()
col_model = col_model.cuda()


def add_label(track_data):
    for track_id in track_data.keys():
        
        pass
    pass

def main(cfg):
    for mode in ['main', 'test']:
        feat_dir = cfg[mode]['feat_dir']
        for track_path in os.listdir(feat_dir):
            full_track_path = osp.join(feat_dir, track_path)
            track_data = pickle_load(full_track_path)
            new_track_data = add_label(track_data)
            pickle_save(new_track_data, full_track_path, verbose=False)
            pass
        pass
    pass

if __name__ == '__main__':
    cfg = {
        'train':{
            'feat_dir': TRAIN_FEAT_DIR,
        },
        'test':{
            'feat_dir': TEST_FEAT_DIR
        }
    }
    main(cfg)
