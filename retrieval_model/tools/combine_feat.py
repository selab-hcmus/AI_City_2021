import os, sys, json, pickle, h5py
import os.path as osp 

list_splits = ['validation', 'train']
ROOT_DIR = '/content/AI_CITY_2021/DATA/data_track_5/AIC21_Track5_NL_Retrieval'
DATA_DIR = osp.join(ROOT_DIR, 'data')
TRAIN_TRACK_JSON = osp.join(DATA_DIR, 'train-tracks.json')
FEAT_SAVE_DIR = '/content/coot-aic/data/aic21'
MODEL_NAME = 'resnet152'
train_track = json.load(open(TRAIN_TRACK_JSON))

def pickle_save(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'save result to {save_path}')

def h5_save(data, save_path):
    data_h5 = h5py.File(save_path, "w")

    for qid in data:
        data_h5[qid] = data[qid]

    data_h5.close()
    print(f'save result to {save_path}')

def pickle_load(save_path):
    data = None
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

total_data = {}

print('>>> combine feats')
for split in list_splits:
    split_name = f'train_{MODEL_NAME}_6148_{split}.pkl'
    split_path = osp.join(FEAT_SAVE_DIR, split_name)
    split_data = pickle_load(split_path)
    total_data.update(split_data)


print('>>> Check')
fail_ids = []
for qid in train_track.keys():
    if total_data.get(qid) is None:
        fail_ids.append(qid)

if len(fail_ids) > 0:
    print(f'{len(fail_ids)} Fail ids')
else:
    pkl_save_path = osp.join(FEAT_SAVE_DIR, f'train_{MODEL_NAME}_6148.pkl')
    h5_save_path = pkl_save_path.replace('.pkl', '.h5')
    pickle_save(total_data, pkl_save_path)
    h5_save(total_data, h5_save_path)
    pass