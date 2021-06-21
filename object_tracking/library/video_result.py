import cv2 
import os 
import os.path as osp
import numpy as np

from utils.data_manager import DATA_DIR
from utils import dict_save, dict_load
from .track_result import TrackResult

from utils import create_logger
debug_logger = create_logger(
    '/home/ntphat/projects/AI_City_2021/results/object_tracking/class_name.log', 
    stdout=False
)


BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
YELLOW = (0, 255, 255)

class VideoResult(object):
    def __init__(self, inp_data) -> None:
        super().__init__()
        self._setup(inp_data)
        pass
    
    ## SETUP
    def _setup(self, inp_data):
        if isinstance(inp_data, str):
            data = dict_load(inp_data)
            vid_id = inp_data.split('.')[0].split('/')[-1]
            self.vid_id = vid_id
        elif isinstance(inp_data, dict):
            data = inp_data
            self.vid_id = None

        self.attn_mask = None
        self.list_frames = data['list_frames']
        self.n_frames = data['n_frames']
        self.frame_ids = data['frame_ids']
        self.n_tracks = data['n_tracks']
        
        self.subject = data.get('subject', None)
        self.stop_tracks = data.get('stop_tracks', [])

        self.track_map = {}
        # Init track result
        for track_id, track_info in data['track_map'].items():
            self.track_map[track_id] = TrackResult(track_id, track_info)
        
        self._set_cam_info()
        pass

    def _set_cam_info(self):
        first_frame = self.list_frames[0] # Ex: "validation/S02/c006/img1/000520.jpg"
        cv_frame = cv2.imread(osp.join(DATA_DIR, first_frame))
        H, W, C = cv_frame.shape
        cam_infos = first_frame.split('/')
        cam_id, street_id = cam_infos[2], cam_infos[1]
        self.cam_id = cam_id
        self.street_id = street_id
        self.height = H
        self.width = W
        pass

    def set_attn_mask(self, mask_path: str):
        self.attn_mask = mask_path

    def set_class_names(self, classifier_manager, veh_thres=0.6, col_thres=0.3):
        debug_logger.info('='*20 +  f' Video {self.vid_id} result ' + '='*20)
        for track_id in self.track_map:
            self.track_map[track_id].set_veh_class(self.list_frames, classifier_manager, veh_thres)
            self.track_map[track_id].set_col_class(self.list_frames, classifier_manager, col_thres)

            # if self.track_map[track_id].is_subject:
            #     print(f'Subject: {track_id}')
            #     # print(self.list_frames)
            #     self.track_map[track_id].set_col_class(self.list_frames, classifier_manager, col_thres)
        pass

    ## MAIN FUNCTIONS
    def _find_subject(self):
        pass 

    def _find_stop_tracks(self):
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
    
    def set_default_subject(self, list_boxes: list, track_id: str):
        track_info = {}
        track_info['boxes'] = list_boxes
        N = len(self.list_frames)
        track_info['frame_order'] = list(range(N))
        track_info['features'] = None
        
        track_res = TrackResult(track_id, track_info)
        self.track_map[track_id] = track_res
        self.set_subject(track_id)
        pass

    def set_subject(self, track_id):
        self.track_map[track_id].is_subject = True
        self.subject = track_id
        pass
    
    def set_stop_tracks(self, stop_tracks: list):
        self.stop_tracks = stop_tracks
        pass

    def get_subject(self):
        return self.track_map[self.subject]

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
                
                # if track_data.flag_thres is not None:
                #     box_name = str(track_data.flag_thres) + "_" + box_name
                
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

    def to_json(self, save_path: str=None, is_feat=False):
        if is_feat:
            save_path = save_path.replace('.json', '.pkl')          

        res = {}
        res['list_frames'] = self.list_frames 
        res['n_frames'] = self.n_frames
        res['frame_ids'] = self.frame_ids 
        res['n_tracks'] = self.n_tracks
        res['subject'] = self.subject
        res['stop_tracks'] = self.stop_tracks
        
        res['height'] = self.height
        res['width'] = self.width
        res['cam_id'] = self.cam_id
        res['street_id'] = self.street_id
        res['attn_mask'] = self.attn_mask

        res['track_map'] = {}
        for track_id in self.track_map:
            res['track_map'][track_id] = self.track_map[track_id].to_json(is_feat)
            pass

        if save_path is not None:
            dict_save(res, save_path)
            pass
        
        return res