from utils.data_manager import json_load
import os, sys, pickle, json
import os.path as osp 
import pandas as pd 
from tqdm import tqdm

from utils import (
    json_save
)


TRAIN_TRACKING_RESULT = './results/annotate_time_train'
TEST_TRACKING_RESULT = './results/annotate_time_test'
MAP_DIR = '../dataset/map_id'

TEST_TRACK_CSV = osp.join(MAP_DIR, 'test_tracks_order.csv')
TRAIN_TRACK_CSV = osp.join(MAP_DIR, 'train_tracks_order.csv')
TEST_QUERY_CSV = osp.join(MAP_DIR, 'test_queries_order.csv')

def create_map_dict(df_map, mode='o2i'):
    map_dict = {} 
    df_key = df_map['key'].values.tolist()
    df_order = df_map['order'].values.tolist()


    if mode == 'i2o':
        for key, order in zip(df_key, df_order):
            map_dict[key] = order
            pass

    if mode == 'o2i':
        for order, key in zip(df_order, df_key):
            map_dict[order] = key
            pass
        
    return map_dict


def get_all_map_dict():
    df_test_track = pd.read_csv(TEST_TRACK_CSV)
    df_test_query = pd.read_csv(TEST_QUERY_CSV)
    df_train_track = pd.read_csv(TRAIN_TRACK_CSV)

    mode = 'o2i'
    dict_test_track = create_map_dict(df_test_track, mode)
    dict_test_query = create_map_dict(df_test_query, mode)
    dict_train_track = create_map_dict(df_train_track, mode)

    json_save(dict_test_track, osp.join(MAP_DIR, f'test_tracks_{mode}.json'))
    json_save(dict_test_query, osp.join(MAP_DIR, f'test_queries_{mode}.json'))
    json_save(dict_train_track, osp.join(MAP_DIR, f'train_tracks_{mode}.json'))
    
    return dict_test_track, dict_test_query, dict_train_track

from data_manager import (
    DATA_DIR, 
    TRAIN_TRACK_JSON, TEST_TRACK_JSON, TEST_QUERY_JSON,
    train_track_map, test_track_map, test_query_map
)
def convert_labels():
    save_name = ['train-tracks_order.json', 'test-tracks_order.json', 'test-queries_order.json']
    path = [TRAIN_TRACK_JSON, TEST_TRACK_JSON, TEST_QUERY_JSON]
    list_maps = [train_track_map, test_track_map, test_query_map]
    save_dir = osp.join(DATA_DIR, 'data')
    
    for i in tqdm(range(3)):
        print(save_name[i])
        raw_data = json_load(path[i])
        new_data = {} 
        map_data = list_maps[i]

        for k in raw_data:
            new_k = map_data[k]
            new_data[new_k] = raw_data[k]
        
        with open(osp.join(save_dir, save_name[i]), 'w') as f:
            json.dump(new_data, f, indent=2)
    
    print(f'save result to {save_dir}')
    pass

if __name__ == '__main__':
    convert_labels()
    # get_all_map_dict()
    pass