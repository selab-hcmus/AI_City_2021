"""Class to handle tracking result

Input: pkl file created by convert_video_track (object_tracking/tools/convert_track.py)
"""
import cv2 
import os 
import os.path as osp
import numpy as np
from pandas.core import frame

from dataset.data_manager import json_load, DATA_DIR
from utils import pickle_load

from object_tracking.utils import (
    get_veh_detection, get_col_detection
)

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)

class TrackResult(object):
    def __init__(self, track_id: str, track_info: dict) -> None:
        super().__init__()
        self.track_id = track_id
        self.frame_order = track_info['frame_order']
        self.first_frame = track_info['frame_order'][0]
        self.last_frame = track_info['frame_order'][-1]
        self.boxes = track_info['boxes']
        self.vehicle_type = track_info['vehicle_type']
        self.color = track_info['color']
        self.cv_boxes = None
        self.features = []
        if track_info.get('features') is not None:
            self.features = track_info['features']
        self.is_subject=False

        self.final_vehicle = None
        self.final_color = None
        pass
    pass

    def set_boxes_data(self, list_frames: list, ids_to_use: list = None):
        cv_boxes = []
        # if ids_to_use is not None:
        #     list_frame = [self.frame_order[i] for i in ids_to_use]
        # else:
        #     list_frame = self.frame_order
        for j, order in enumerate(ids_to_use):
            fpath = osp.join(DATA_DIR, list_frames[self.frame_order[order]])
            cv_img = cv2.imread(fpath)
            box = self.boxes[order] #xyxy
            box = [int(i) for i in box]
            cv_box = cv_img[box[1]:box[3], box[0]: box[2], :]
            cv_boxes.append(cv_box)

        return cv_boxes

    def get_valid_index(self, idx, max_idx):
        return max(min(idx, max_idx), 0)

    def set_veh_class(self, list_frames, classifier_manager, thres=0.7):
        N = len(self.boxes)
        # ids_to_use = [N//6, 2*N//6, 3*N//6, 4*N//6, 5*N//6]
        ids_to_use = [N//2-2, N//2-1, N//2, N//2+1, N//2+2]
        ids_to_use = [self.get_valid_index(i, N-1) for i in ids_to_use]
        weights = np.ones(len(ids_to_use))
        weights[1:4] = 2

        boxes_to_use = self.set_boxes_data(list_frames, ids_to_use)
        box_names, self.final_vehicle, _, _ = classifier_manager.get_veh_predictions(boxes_to_use, thres, weights)
        pass 

    def get_final_classname(self):
        res = None
        if self.final_color is None and self.final_vehicle is not None:
            res = self.final_vehicle
        
        if self.final_color is not None and self.final_vehicle is not None:
            res = f'{self.final_color}-{self.final_vehicle}'
        
        return res
    
    def interpolate_frames(self):
        # TODO
        pass


class VideoResult(object):
    def __init__(self, save_path: str) -> None:
        super().__init__()
        self._setup(save_path)
        pass
    
    ## SETUP
    def _setup(self, save_path):
        if '.pkl' in save_path:
            data = pickle_load(save_path)
        else:
            data = json_load(save_path)

        self.list_frames = data['list_frames']
        self.n_frames = data['n_frames']
        self.frame_ids = data['frame_ids']
        self.n_tracks = data['n_tracks']
        self.track_map = {}

        self.subject = None
        self.stop_tracks = []
        if data.get('subject') is not None:
            self.subject = data['subject']
        if data.get('stop_tracks') is not None:
            self.stop_tracks = data['stop_tracks']
        
        # Init track result
        for track_id, track_info in data['track_map'].items():
            self.track_map[track_id] = TrackResult(track_id, track_info)
        pass

    def set_class_names(self, classifier_manager):
        for track_id in self.track_map:
            # self.track_map[track_id].set_boxes_data(self.list_frames)
            self.track_map[track_id].set_veh_class(self.list_frames, classifier_manager, 0.5)
        pass

    ## UTILITIES
    def get_list_tracks(self):
        return list(self.track_map.values())
    
    def get_list_tracks_by_time(self):
        def sort_func(x: TrackResult):
            return x.first_frame

        list_tracks = list(self.track_map.values())
        list_tracks.sort(key=sort_func)
        return list_tracks
    
    def add_new_track(self, track_data: TrackResult, track_id: str):
        if self.track_map.get(track_id) is not None:
            print(f'Track id {track_id} existed')
            return
        self.track_map[track_id] = track_data
        pass 

    def remove_track(self, track_id: str):
        self.track_map.pop(track_id, None)
        pass
    
    def set_subject(self, track_id):
        for tid in self.track_map:
            if tid == track_id:
                self.track_map[tid].is_subject = True
                break
        self.subject = self.track_map[tid]
        pass

    def get_subject(self):
        return self.subject

    def visualize(self, save_path: str, attn_mask: np.ndarray=None):
        list_frames = []
        for fname in self.list_frames:
            fpath = osp.join(DATA_DIR, fname)
            list_frames.append(cv2.imread(fpath))

        vid_height, vid_width, _ = list_frames[0].shape
        
        for track_id in self.track_map:
            is_subject = (track_id == self.subject)
            is_stop = (track_id in self.stop_tracks)

            track_data = self.track_map[track_id]

            for i, frame_order in enumerate(track_data.frame_order):
                cv_frame = list_frames[frame_order]
                bbox = track_data.boxes[i] #xyxy

                box_name = f'{track_id}'
                vis_color = GREEN
                if track_data.get_final_classname() is not None:
                    box_name += f'_{track_data.get_final_classname()}'

                if is_subject:
                    box_name += '_sb'
                    vis_color = RED

                if is_stop:
                    box_name += '_stop'
                    vis_color = BLUE
                
                cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),vis_color, 2)
                cv2.putText(cv_frame, box_name,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 150, vis_color, 2)

                pass
            pass

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_writer = cv2.VideoWriter(save_path,fourcc, 1, (vid_width, vid_height))
        for cv_frame in list_frames:
            if attn_mask is not None:
                attn_mask[np.where(attn_mask <= 0.3)] = 0.3
                cv_frame = (cv_frame*attn_mask).astype(np.uint8)

            vid_writer.write(cv_frame)
            pass
        vid_writer.release()

        pass