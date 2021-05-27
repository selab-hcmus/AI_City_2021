import os, sys, pickle, json
import os.path as osp 
import pandas as pd 

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

if __name__ == '__main__':
    get_all_map_dict()
    pass