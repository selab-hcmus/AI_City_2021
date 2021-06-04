import json, pickle 

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
