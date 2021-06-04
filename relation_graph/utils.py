import os
import os.path as osp
import pickle 
import cv2

class PositionState:
    NO_RELATION = 0
    A_BEHIND_B = 1
    B_BEHIND_A = 2
    pass

class FollowState:
    NO_RELATION = 0
    A_FOLLOW_B = 1
    B_FOLLOW_A = 2
    RELATION_NAME = {
        0: 'NO_RELATION', 1: 'A_FOLLOW_B', 2: 'B_FOLLOW_A'
    }
    pass

NUM_COUNT_THRES = 9

class Counter(object):
    def __init__(self):
        self.counter = {}
        self.total = 0
        self.famous_value = None
        self.max_count = 1
        pass
    
    def update(self, value):
        if self.counter.get(value) is None:
            self.counter[value] = 1
        else:
            self.counter[value] += 1

        if self.counter[value] > self.max_count:
            self.max_count = self.counter[value]
            self.famous_value = value 

        self.total += 1
    
    def get_famous_value(self):
        a_fl_b = self.counter.get(FollowState.A_FOLLOW_B, None)
        b_fl_a = self.counter.get(FollowState.B_FOLLOW_A, None)
        if a_fl_b is None and b_fl_a is None:
            return FollowState.NO_RELATION
        elif a_fl_b is None:
            if b_fl_a >= NUM_COUNT_THRES:
                return FollowState.B_FOLLOW_A
            return FollowState.NO_RELATION
        elif b_fl_a is None:
            if a_fl_b >= NUM_COUNT_THRES:
                return FollowState.A_FOLLOW_B
            return FollowState.NO_RELATION
        else:
            if a_fl_b >= b_fl_a:
                return FollowState.A_FOLLOW_B
            return FollowState.A_FOLLOW_B
        # return self.famous_value
        
    pass

def smooth_distance(list_dis: list, skip_frame: int = 5):
    if len(list_dis) < skip_frame:
        return list_dis
    
    pass

def minus_vector(vector_a, vector_b):
    return [vector_b[0] - vector_a[0], vector_b[1] - vector_a[1]] 
    
def calculate_velocity_vector(coor: list, skip_frame=2, smooth_frame=2):
    if skip_frame > len(coor):
        skip_frame = 1
    vel_list = [minus_vector(coor[i], coor[i+skip_frame]) for i in range(len(coor) - skip_frame)]
    
    return vel_list

def xyxy_to_xywh(coor):
    x1, y1, x2, y2 = coor
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]


def visualize(json_data: dict, list_track_ids: list, data_dir: str, vid_save_path: str):
    list_frames = [] 

    for frame_name in json_data['list_frames']:
        cv_frame = cv2.imread(osp.join(data_dir, frame_name))
        list_frames.append(cv_frame)
        pass
    
    vid_height, vid_width, _ = list_frames[0].shape
    for track_id in list_track_ids:
        track_data = json_data['track_map'][track_id]
        for i in range(len(track_data['boxes'])):
            frame_order = track_data['frame_order'][i]
            bbox = track_data['boxes'][i]
            cv_frame = list_frames[frame_order]

            cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(cv_frame, str(track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        pass
    pass

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(vid_save_path,fourcc, 1, (vid_width, vid_height))
    for cv_frame in list_frames:
        vid_writer.write(cv_frame)
        pass
    vid_writer.release()
    pass