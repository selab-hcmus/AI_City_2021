import os, sys, pickle, json
import os.path as osp 
import pandas as pd 


TRAIN_TRACKING_RESULT = './results/annotate_time_train'
TEST_TRACKING_RESULT = './results/annotate_time_test'
MAP_DIR = '../dataset/map_id'

TEST_TRACK_CSV = osp.join(MAP_DIR, 'test_tracks_order.csv')
TRAIN_TRACK_CSV = osp.join(MAP_DIR, 'train_tracks_order.csv')
TEST_QUERY_CSV = osp.join(MAP_DIR, 'test_queries_order.csv')

def create_map_dict(df_map):
    map_dict = {} 
    df_key = df_map['key'].values.tolist()
    df_order = df_map['order'].values.tolist()
    for key, order in zip(df_key, df_order):
        map_dict[key] = order
        pass
    
    return map_dict

def json_save(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    pass

def get_all_map_dict():
    df_test_track = pd.read_csv(TEST_TRACK_CSV)
    df_test_query = pd.read_csv(TEST_QUERY_CSV)
    df_train_track = pd.read_csv(TRAIN_TRACK_CSV)

    dict_test_track = create_map_dict(df_test_track)
    dict_test_query = create_map_dict(df_test_query)
    dict_train_track = create_map_dict(df_train_track)

    json_save(dict_test_track, osp.join(MAP_DIR, 'test_tracks.json'))
    json_save(dict_test_query, osp.join(MAP_DIR, 'test_queries.json'))
    json_save(dict_train_track, osp.join(MAP_DIR, 'train_tracks.json'))
    
    return dict_test_track, dict_test_query, dict_train_track

if __name__ == '__main__':
    get_all_map_dict()
    pass