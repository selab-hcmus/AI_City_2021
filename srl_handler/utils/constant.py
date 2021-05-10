import os, json 
import os.path as osp

# WORKING_DIR = '/Users/ntphat/Documents/THESIS/SOURCE/aic2021'
# DATA_DIR = '/Volumes/TIENPHAT/THESIS/data_aic2021/data' 
# RESULT_DIR = osp.join(WORKING_DIR, 'results') 

DATA_DIR = '../dataset' 

TEST_QUERY_JSON = '../dataset/data/test-queries.json'
TEST_TRACK_JSON = '../dataset/data/test-tracks.json'
TRAIN_TRACK_JSON = '../dataset/data/train-tracks.json'

# REFINED_TRAIN_TRACK_JSON = osp.join(WORKING_DIR, 'results/refined_boxes/refined_train_tracks_Centernet2_v2.json') 
REFINED_TEST_TRACK_JSON = ''

VEHICLE_VOCAB_JSON = './data/vehicle_vocabulary.json'
COLOR_VOCAB_JSON = './data/color_vocabulary.json'
ACTION_VOCAB_JSON = './data/action_vocabulary.json'

VEHICLE_GROUP_JSON = './data/vehicle_group_v1.json'
VEHICLE_GROUP_REP_JSON = VEHICLE_GROUP_JSON.replace('.json', '_rep.json')
COLOR_GROUP_JSON = './data/color_group_v1.json'
ACTION_GROUP_JSON = './data/action_group_v1.json'

# Preprocess result
TRAIN_SRL_JSON = '../srl_extraction/results/result_train.json' #osp.join(WORKING_DIR, 'results/preprocess_text/result_train_beta_v2.json') 
TEST_SRL_JSON = '../srl_extraction/results/result_test.json'

VEHICLE_VOCAB = json.load(open(VEHICLE_VOCAB_JSON, 'r'))
COLOR_VOCAB = json.load(open(COLOR_VOCAB_JSON, 'r'))
ACTION_VOCAB = json.load(open(ACTION_VOCAB_JSON, 'r'))
# # COLOR_GROUP = json.load(open(COLOR_GROUP_JSON, 'r'))
VEHICLE_GROUP_REP = json.load(open(VEHICLE_GROUP_REP_JSON, 'r'))
LIST_REDUNDANT_VEHICLES = ['volvo', 'chevrolet', 'vehicle', 'car']
