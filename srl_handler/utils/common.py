import cv2 
import os 
import json
from tqdm import tqdm 
import os.path as osp 

from .constant import (
    VEHICLE_GROUP_REP, VEHICLE_VOCAB, LIST_REDUNDANT_VEHICLES, COLOR_VOCAB, ACTION_VOCAB
)

def remove_redundant_actions(list_actions: list):
    res = [c for c in list_actions if c in ACTION_VOCAB]
    return res

def remove_redundant_colors(list_colors: list):
    res = [c for c in list_colors if c in COLOR_VOCAB]
    return res

def remove_redundant_subjects(list_subjects: list):
    res = [s for s in list_subjects if s not in LIST_REDUNDANT_VEHICLES]
    return res

def convert_to_representation_subject(list_subjects: list):
    map_dict = {}
    for k, v in VEHICLE_GROUP_REP.items():
        for veh in v:
            map_dict[veh] = k
    
    res = []
    fail = []
    for s in list_subjects:
        if map_dict.get(s) is None:
            fail.append(s)
            continue
        res.append(map_dict[s])
    
    if len(fail):
        print(fail)
    return res

def is_list_in_list(list_values, list_to_check):
    res = True 
    for val in list_to_check:
        if val not in list_values:
            return False
    return True

def get_vehicle_name_map(vehicle_vocab: dict):
    """Convert from {"group-1": SUV} to {"SUV": 1}
    """
    id_map = {} #{'group-1': 0}
    for k in vehicle_vocab.keys():
        i = int(k.split('-')[1]) - 1
        id_map[k] = i

    veh_map = {} # {'suv': 2}
    for k in vehicle_vocab.keys():
        i = id_map[k]
        for veh in vehicle_vocab[k]:
            veh_map[veh] = i 
    
    return veh_map

def dump_json(data_dict, json_path, verbose=False):
    with open(json_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    if verbose:
        print(f'Save result to {json_path}')

def scan_images(list_img_path):
    for img_path in tqdm(list_img_path):
        pass
    pass
