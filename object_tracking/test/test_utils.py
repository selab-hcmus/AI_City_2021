import json 

def xyxy_to_xywh(box):
    return [box[0], box[1], box[2]-box[0], box[3] - box[1]]

def json_save(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    pass

def json_load(save_path):
    data = None
    with open(save_path, 'r') as f:
        data = json.load(f)
    return data
