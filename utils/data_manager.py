import os 
import os.path as osp 
import json 

# DATA_DIR = '/content/AI_City_2021/dataset'
# RESULT_DIR = '/content/AI_City_2021/results'

DATA_DIR = '/home/ntphat/projects/AI_City_2021/dataset'
RESULT_DIR = '/home/ntphat/projects/AI_City_2021/results'

TRAIN_TRACK_JSON = osp.join(DATA_DIR, 'data/train-tracks.json')
TEST_TRACK_JSON = osp.join(DATA_DIR, 'data/test-tracks.json')
TEST_QUERY_JSON = osp.join(DATA_DIR, 'data/test-queries.json')

TRAIN_TRACK_ORDER_JSON = osp.join(DATA_DIR, 'data/train-tracks_order.json')
TEST_TRACK_ORDER_JSON = osp.join(DATA_DIR, 'data/test-tracks_order.json')
TEST_QUERY_ORDER_JSON = osp.join(DATA_DIR, 'data/test-queries_order.json')

MAP_ID_DIR = osp.join(DATA_DIR, 'map_id')
TEST_TRACK_MAP_JSON = osp.join(MAP_ID_DIR, 'test_tracks.json')
TEST_QUERY_MAP_JSON = osp.join(MAP_ID_DIR, 'test_queries.json')
TRAIN_TRACK_MAP_JSON = osp.join(MAP_ID_DIR, 'train_tracks.json')

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data

def setup_reverse_map(map_dict: dict):
    order_id_map = {}
    for k, v in map_dict.items():
        order_id_map[str(v)] = k 
    
    map_dict.update(order_id_map)
    return map_dict

test_track_map = json_load(TEST_TRACK_MAP_JSON)
test_query_map = json_load(TEST_QUERY_MAP_JSON)
train_track_map = json_load(TRAIN_TRACK_MAP_JSON)
test_track_map = setup_reverse_map(test_track_map)
test_query_map = setup_reverse_map(test_query_map)
train_track_map = setup_reverse_map(train_track_map)
