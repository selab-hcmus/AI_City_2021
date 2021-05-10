import os, json, cv2, sys
import os.path as osp
import pandas as pd 
from tqdm import tqdm
import numpy as np 
sys.path.append('./preprocess')

from library.text.query import Query

from utils.constant import (
    TRAIN_SRL_JSON, TEST_SRL_JSON, TRAIN_TRACK_JSON, TEST_TRACK_JSON,
    VEHICLE_VOCAB, COLOR_VOCAB, COLOR_GROUP_JSON, DATA_DIR, ACTION_GROUP_JSON, ACTION_VOCAB_JSON
)

pd.set_option('display.max_columns', None)

IS_TEST = True
TYPE = 'action'
SAVE_DIR = './results'
data_group = json.load(open(ACTION_GROUP_JSON, 'r'))
num_classes = len(data_group.keys())

id_map = {} #{'group-1': 0}
for k in data_group.keys():
    i = int(k.split('-')[1]) - 1
    id_map[k] = i

veh_map = {} # {'suv': 2}
for k in data_group.keys():
    i = id_map[k]
    for veh in data_group[k]:
        veh_map[veh] = i 
print(veh_map)
def create_ohe_vector(list_vehicles, use_fraction=False):
    y  = np.zeros(num_classes)
    flag = True #Check if exist at least one valid vehicle or not
    for veh in list_vehicles:
        if veh_map.get(veh) is None:
            print(f'invalid action: {veh}')
            continue
        flag = False
        if use_fraction:
            y[veh_map[veh]] += 1
        else:
            y[veh_map[veh]] = 1
    
    if flag:
        return None

    if use_fraction:
        y /= np.sum(y)

    return y

def parse_to_csv(data_srl, mode='test', use_fraction=True, is_csv=True):
    df_dict = {
        'query_id': [], 'captions': [], 'actions': [], 'labels': [],
    }
    
    fail_query_ids = []
    for query_id in tqdm(data_srl.keys()):
        query_content = data_srl[query_id]
        query = Query(query_content, query_id)
    
        query.actions.sort()
        query_veh_labels = create_ohe_vector(query.actions, use_fraction)
        if query_veh_labels is None:
            fail_str = f'{query_id}: {query.actions}'
            fail_query_ids.append(fail_str)
            continue
        
        df_dict['query_id'].append(query_id)
        df_dict['labels'].append(query_veh_labels.tolist())
        df_dict['actions'].append(query.actions)
        list_caps = [c.caption for c in query.list_caps]
        df_dict['captions'].append('\n'.join(list_caps))

    df_final = None
    if is_csv:
        df_final = pd.DataFrame.from_dict(df_dict)
        csv_save_path = osp.join(SAVE_DIR, f'{TYPE}_{mode}_ohe.csv')
        if use_fraction is True:
            csv_save_path = osp.join(SAVE_DIR, f'{TYPE}_{mode}_fraction.csv')

        df_final.to_csv(csv_save_path, index=False)

        print(f'save result to {SAVE_DIR} directory')
        print(f'{len(fail_query_ids)} fail queries: {fail_query_ids}')
    return df_final


if __name__ == '__main__':
    PRINT_CSV = True
    test_srl = json.load(open(TEST_SRL_JSON))

    parse_to_csv(test_srl, 'test', use_fraction=True, is_csv=PRINT_CSV)
    parse_to_csv(test_srl, 'test', use_fraction=False, is_csv=PRINT_CSV)
