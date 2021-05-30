import json 

def json_save(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    pass
