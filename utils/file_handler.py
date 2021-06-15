import os
import json, pickle 

def prepare_dir(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def dict_save(data, save_path: str):
    if '.json' in save_path:
        json_save(data, save_path)
    elif '.pkl' in save_path:
        pickle_save(data, save_path)
    else:
        file_format = save_path.split('.')[-1]
        print(f'Not supported format {file_format} yet')
    pass

def dict_load(save_path: str):
    if '.json' in save_path:
        return json_load(save_path)
    elif '.pkl' in save_path:
        return pickle_load(save_path)
    else:
        file_format = save_path.split('.')[-1]
        print(f'Not supported format {file_format} yet')
    pass


def json_save(data, save_path, verbose=False):
    if verbose:
        print(f'Save data to {save_path}')
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    pass

def json_load(save_path, verbose=False):
    if verbose:
        print(f'Load data from {save_path}')
    data = None
    with open(save_path, 'r') as f:
        data = json.load(f)
    return data

def pickle_load(save_path, verbose=False):
    if verbose:
        print(f'Load data from {save_path}')
    data=None 
    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    return data

def pickle_save(data, save_path, verbose=False):
    if verbose:
        print(f'Save data to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    pass
