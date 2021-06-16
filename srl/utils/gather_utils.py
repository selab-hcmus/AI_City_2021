from utils import json_load
import numpy as np

def setup_info(data_info, data_json):
    num_classes, label_map = get_label_info(data_json)
    data_info['num_classes'] = num_classes
    data_info['label_map'] = label_map
    pass

def get_label_info(label_group_json: str):
    label_group = json_load(label_group_json)
    num_classes = len(label_group.keys())

    id_map = {} #{'group-1': 0}
    for k in label_group.keys():
        i = int(k.split('-')[1]) - 1
        id_map[k] = i

    label_map = {} # {'suv': 2}
    for k in label_group.keys():
        i = id_map[k]
        for veh in label_group[k]:
            label_map[veh] = i 
    
    return num_classes, label_map

def get_label_vector(list_values, num_classes, label_map, is_test=True, use_fraction=False):
    y  = np.zeros(num_classes)
    flag = True #Check if exist at least one valid vehicle or not
    for val in list_values:
        if label_map.get(val) is None:
            # print(f'invalid value: {val}')
            continue
        flag = False
        if use_fraction:
            y[label_map[val]] += 1
        else:
            y[label_map[val]] = 1
    
    if flag:
        if is_test:
            return np.ones(num_classes).tolist()
        else:
            return None

    if use_fraction:
        y /= np.sum(y)

    return y.tolist()


