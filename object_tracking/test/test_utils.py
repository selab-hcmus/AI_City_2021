import json 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def xywh_to_xyxy(box):
    return [box[0], box[1], box[2] + box[0], box[3] + box[1]]

def xyxy_to_xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

def json_save(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    pass

def json_load(save_path):
    data = None
    with open(save_path, 'r') as f:
        data = json.load(f)
    return data

def a_substract_b(list_a: list, list_b: list):
    return [ i for i in list_a if i not in list_b]
    
def is_miss_frame(track_data: dict):
    prev = int(track_data['frame_order'][0])
    for frame_order in track_data['frame_order'][1:]:
        order = int(frame_order)
        if order - prev != 1:
            return True
        prev = order
        pass
    return False

def get_miss_frame_tracks(vid_data: dict):
    fail_tracks = []
    for track_id in vid_data['track_map']:
        if is_miss_frame(vid_data['track_map'][track_id]):
            fail_tracks.append(track_id)
        pass
    return fail_tracks