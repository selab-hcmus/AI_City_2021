import os, json, cv2, sys, shutil
import os.path as osp
import pandas as pd 
from tqdm import tqdm
import numpy as np 
sys.path.append('./preprocess')

from library.text.query import Query
from utils.constant import (
    TRAIN_SRL_JSON, TEST_SRL_JSON, TRAIN_TRACK_JSON, TEST_TRACK_JSON, VEHICLE_GROUP_JSON,
    VEHICLE_VOCAB, COLOR_VOCAB, DATA_DIR
)

pd.set_option('display.max_columns', None)

TYPE = 'veh'
BOX_FIELD = 'boxes'
SAVE_DIR = './results'
vehicle_group_json = VEHICLE_GROUP_JSON
vehicle_group = json.load(open(vehicle_group_json, 'r'))
num_classes = len(vehicle_group.keys())
id_map = {} #{'group-1': 0}

for k in vehicle_group.keys():
    i = int(k.split('-')[1]) - 1
    id_map[k] = i

veh_map = {} # {'suv': 2}
for k in vehicle_group.keys():
    i = id_map[k]
    for veh in vehicle_group[k]:
        veh_map[veh] = i 

def create_ohe_vector(list_vehicles, use_fraction=False):
    y  = np.zeros(num_classes)
    flag = True #Check if exist at least one valid vehicle or not
    for veh in list_vehicles:
        if veh_map.get(veh) is None:
            print(f'invalid veh: {veh}')
            continue
        flag = False
        if use_fraction:
            y[veh_map[veh]] += 1
        else:
            y[veh_map[veh]] = 1
    
    if flag:
        return np.ones(num_classes) 

    if use_fraction:
        y /= np.sum(y)

    return y

def get_list_boxes(data_track, query_id, save_dir=None):
    list_boxes = data_track[query_id][BOX_FIELD]
    n = len(list_boxes)
    ids2use = [0, n//3, 2*n//3, n-1]

    res = {'paths': [], 'width': [], 'height': []}
    for i in ids2use:
        img_path = osp.join(DATA_DIR, data_track[query_id]['frames'][i])
        cv_img = cv2.imread(img_path)
        x, y, w, h = list_boxes[i]
        cv_box = cv_img[y:y+h, x:x+w, :]
        res['width'].append(w)
        res['height'].append(h)
        
        if save_dir is not None:
            box_save_path = osp.join(save_dir, f'{query_id}_{i}.png')
            res['paths'].append(box_save_path)

            if not osp.isfile(box_save_path):
                cv2.imwrite(box_save_path, cv_box)
            
    return res

def get_manual_labels(query_id):
    label = df_manual[df_manual['query_id'] == query_id]['new_labels']
    if isinstance(label, str):
        label = eval(label)
    return label

def parse_to_csv(data_srl, data_track, mode='train', fraction=True):
    df_dict = {
        'query_id': [], 'box_id': [], 'width': [], 'height': [], 
        'labels': [], 'vehicles': [], 'vehicle_code': [], 'paths': []
    }
    box_id = 0
    train_vis_dir = osp.join(SAVE_DIR, 'veh_imgs')
    os.makedirs(train_vis_dir, exist_ok=True)

    fail_query_ids = []
    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)
        query.subjects.sort()

        query_veh_labels = create_ohe_vector(query.subjects, use_fraction=fraction)
        if query_veh_labels is None:
            fail_query_ids.append(query_id)
            continue

        res = get_list_boxes(data_track, query_id, train_vis_dir)
        for i in range(len(res['width'])):
            box_id += 1
            df_dict['query_id'].append(query_id)
            df_dict['labels'].append(query_veh_labels.tolist())
            df_dict['vehicles'].append(query.subjects)
            
            # Box information
            df_dict['box_id'].append(box_id)
            df_dict['width'].append(res['width'][i])
            df_dict['height'].append(res['height'][i])
            df_dict['paths'].append(res['paths'][i])
    

    df_final = pd.DataFrame.from_dict(df_dict)
    csv_save_path = osp.join(SAVE_DIR, f'{mode}.csv')
    if fraction is True:
        csv_save_path = osp.join(SAVE_DIR, f'{mode}_fraction.csv')

    df_final.to_csv(csv_save_path, index=False)
    print(f'Extract dataframe to {csv_save_path}')    
    print(f'Fail queries: {fail_query_ids}')
    return df_final

def parse_to_csv_test(data_srl, data_track, mode='test', fraction=True):
    df_dict = {
        'query_id': [], 'queries': [], 'vehicles': [], 'labels': [], 
    }
    box_id = 0
    
    fail_query_ids = []
    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)
        query.subjects.sort()

        query_veh_labels = create_ohe_vector(query.subjects, use_fraction=fraction)
        if query_veh_labels is None:
            fail_str = f'{query_id}: {query.subjects}'
            fail_query_ids.append(fail_str)
            continue
        
        df_dict['queries'].append(query.get_list_captions_str())
        df_dict['query_id'].append(query_id)
        df_dict['labels'].append(query_veh_labels.tolist())
        df_dict['vehicles'].append(query.subjects)

    df_final = pd.DataFrame.from_dict(df_dict)
    csv_save_path = osp.join(SAVE_DIR, f'{TYPE}_{mode}.csv')
    if fraction is True:
        csv_save_path = osp.join(SAVE_DIR, f'{TYPE}_{mode}_fraction.csv')

    df_final.to_csv(csv_save_path, index=False)
    print(f'Extract dataframe to {csv_save_path}')    
    print(f'Fail queries: {fail_query_ids}')
    return df_final
    

def main():
    # Train tracks
    print('RUN TRAIN')
    train_srl = json.load(open(TRAIN_SRL_JSON))
    train_track = json.load(open(TRAIN_TRACK_JSON))
    parse_to_csv(train_srl, train_track, 'train', fraction=True)

    # Test tracks
    print('RUN TEST')
    train_srl = json.load(open(TEST_SRL_JSON))
    train_track = json.load(open(TEST_TRACK_JSON))
    parse_to_csv_test(train_srl, train_track, 'test', fraction=True)
    parse_to_csv_test(train_srl, train_track, 'test', fraction=False)

    pass

if __name__ == '__main__':
    main()