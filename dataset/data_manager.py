import os 
import os.path as osp 
import json 

DATA_DIR = '/home/ntphat/projects/AI_City_2021/dataset'
MAP_ID_DIR = osp.join(DATA_DIR, 'map_id')
TEST_TRACK_JSON = osp.join(MAP_ID_DIR, 'test_tracks.json')
TEST_QUERY_JSON = osp.join(MAP_ID_DIR, 'test_queries.json')
TRAIN_TRACK_JSON = osp.join(MAP_ID_DIR, 'train_tracks.json')

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

test_track_map = json_load(TEST_TRACK_JSON)
test_query_map = json_load(TEST_QUERY_JSON)
train_track_map = json_load(TRAIN_TRACK_JSON)
