import os, json 
import os.path as osp

from utils.data_manager import (
    DATA_DIR, RESULT_DIR, TEST_QUERY_JSON, TEST_TRACK_JSON, TRAIN_TRACK_JSON
)
from utils import dict_load

# WORKING_DIR = '/Users/ntphat/Documents/THESIS/SOURCE/aic2021'
# DATA_DIR = '/Volumes/TIENPHAT/THESIS/data_aic2021/data' 
# RESULT_DIR = osp.join(WORKING_DIR, 'results') 

SAVE_DIR = osp.join(RESULT_DIR, 'srl')
SRL_DATA_DIR = osp.join(DATA_DIR, 'srl')

# REFINED_TRAIN_TRACK_JSON = osp.join(WORKING_DIR, 'results/refined_boxes/refined_train_tracks_Centernet2_v2.json') 
REFINED_TEST_TRACK_JSON = ''

VEHICLE_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'vehicle_vocabulary.json')
COLOR_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'color_vocabulary.json') 
ACTION_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'action_vocabulary.json') 

VEHICLE_GROUP_JSON = osp.join(SRL_DATA_DIR, 'vehicle_group_v1.json') 
VEHICLE_GROUP_REP_JSON = VEHICLE_GROUP_JSON.replace('.json', '_rep.json')
COLOR_GROUP_JSON = osp.join(SRL_DATA_DIR, 'color_group_v1.json') 
ACTION_GROUP_JSON = osp.join(SRL_DATA_DIR, 'action_group_v1.json') 

# Preprocess result
TRAIN_SRL_JSON = osp.join(SAVE_DIR, 'result_train.json') 
TEST_SRL_JSON = osp.join(SAVE_DIR, 'result_test.json') 

VEHICLE_VOCAB = dict_load(VEHICLE_VOCAB_JSON) 
COLOR_VOCAB = dict_load(COLOR_VOCAB_JSON) 
ACTION_VOCAB = dict_load(ACTION_VOCAB_JSON) 
# # COLOR_GROUP = json.load(open(COLOR_GROUP_JSON, 'r'))
VEHICLE_GROUP_REP = dict_load(VEHICLE_GROUP_REP_JSON) #json.load(open(VEHICLE_GROUP_REP_JSON, 'r'))
LIST_REDUNDANT_VEHICLES = ['volvo', 'chevrolet', 'vehicle', 'car']
