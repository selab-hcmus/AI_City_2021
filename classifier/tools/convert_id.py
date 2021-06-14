import os
import os.path as osp
from tqdm import tqdm
import json
from utils.data_manager import (
    test_track_map, train_track_map
)

train_feat_dir = '/home/ntphat/projects/AI_City_2021/classifier/results/train_feat_tracking'
test_feat_dir = '/home/ntphat/projects/AI_City_2021/classifier/results/test_feat_tracking'

def convert_feat_id(convert_map: dict, feat_dir: str):
    for fname in tqdm(os.listdir(feat_dir)):
        infos = fname.split('.')
        old_id = infos[0]
        if convert_map.get(old_id) is None:
            continue
        new_id = convert_map[old_id]
        new_fname = f'{new_id}.{infos[1]}'
        os.rename(
            src = osp.join(feat_dir, fname),
            dst = osp.join(feat_dir, new_fname)
        )
    pass

def convert_box_id(convert_map: dict, box_path: str):
    old_data = json.load(open(box_path))
    list_old_keys = list(old_data.keys())
    res = {}
    for k in tqdm(list_old_keys):
        new_k = convert_map[k]
        res[new_k] = old_data[k]
    
    with open(box_path.replace('_boxes', '_order'), 'w') as f:
        json.dump(res, f, indent=2)
    pass

# convert_feat_id(test_track_map, test_feat_dir)
# convert_feat_id(train_track_map, train_feat_dir)

convert_box_id(test_track_map, '/home/ntphat/projects/AI_City_2021/classifier/data/Centernet2_test_veh_boxes.json')
convert_box_id(train_track_map, '/home/ntphat/projects/AI_City_2021/classifier/data/Centernet2_train_veh_boxes.json')

